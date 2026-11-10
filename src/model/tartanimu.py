"""Adapter around the released TartanIMU foundation model vendored at ``thirdparty/tartanimu``:
frozen-checkpoint body-velocity inference on body-frame IMU with gravity kept in the accelerometer."""

import sys

from pathlib import Path
from dataclasses import dataclass

import yaml
import torch
import torch.nn as nn
import pypose as pp
import jaxtyping as jt
import typing as tp

from ..interface.data_imu import IMUData_w_GT_Jagged, GRAVITY_STANDARD

# Appended to sys.path, not prepended: the checkout root holds generic modules (``config``, ``tools``).
_TARTANIMU_ROOT = Path(__file__).resolve().parents[2] / "thirdparty/tartanimu"

# Fixed by the released checkpoint: the trunk's LSTM input width encodes the window length.
NET_WINDOW = 40     # post-decimation samples per window
SEQ_LEN    = 10     # windows per LSTM sequence; only the last one is read out
WINDOW_S   = 1.0    # seconds of real time per window

# ``motion_type`` ids of the official dataloader.
MOTION_TYPE_ID = {"car": 1, "dog": 2, "drone": 3, "human": 4}

# Human motion capture and the bipedal humanoid both read the ``human`` head.
DEFAULT_HEAD = "human"


def _update_recursive(dst: dict, src: dict) -> None:
    """``tartan_imu`` ``update_recursive``, inlined: the vendored configer pulls in torch.distributed
    and the trainer."""
    for key, value in src.items():
        if isinstance(value, dict):
            _update_recursive(dst.setdefault(key, {}), value)
        else:
            dst[key] = value


def _attr(owner: nn.Module, name: str) -> tp.Any:
    """``getattr`` narrowed to ``Any``; ``nn.Module.__getattr__`` is typed ``Tensor | Module``."""
    return getattr(owner, name)


def _build_backbone(cfg: dict) -> nn.Module:
    """Construct the released architecture through the vendored registry."""
    if str(_TARTANIMU_ROOT) not in sys.path:
        sys.path.append(str(_TARTANIMU_ROOT))
    from tartan_imu.model.registry import build_backbone
    return build_backbone(cfg["model"]["model_name"], cfg)


@dataclass
class TartanIMU_Output:
    # Body-frame velocity per prediction, and the stream sample it is anchored to (its window centre).
    vel    : jt.Float[torch.Tensor, "K 3"]
    centers: jt.Int64[torch.Tensor, "K"]


class TartanIMU(nn.Module):
    def __init__(
        self,
        ckpt        : str,
        config      : str,
        model_config: str,
        head        : str = DEFAULT_HEAD,
    ):
        super().__init__()
        assert head in MOTION_TYPE_ID, f"head must be one of {sorted(MOTION_TYPE_ID)}, got {head!r}"

        cfg = yaml.safe_load(Path(config).expanduser().read_text())
        _update_recursive(cfg, yaml.safe_load(Path(model_config).expanduser().read_text()))

        # Only the ``use_local_coord`` arm is reproduced (body frame, gravity kept, window-mean body velocity).
        assert cfg["data"]["use_local_coord"], \
            "this adapter implements TartanIMU's body-frame (use_local_coord) arm; the released unified.yaml sets it"
        param = cfg["model_param"]
        assert param["past_time"] == 0.0 and param["future_time"] == 0.0, \
            f"non-zero past/future context is not modelled here: {param['past_time']} / {param['future_time']}"
        assert param["window_time"] == WINDOW_S, f"expected a {WINDOW_S}s window, config says {param['window_time']}"
        net_window = int(round(param["window_time"] * cfg["data"]["sample_freq"]))
        assert net_window == NET_WINDOW, \
            f"released weights pin the window to {NET_WINDOW} post-decimation samples, config gives {net_window}"

        self.cfg     = cfg
        self.head    = head
        self.net_hz  = float(cfg["data"]["sample_freq"])
        self.net     = _build_backbone(cfg)

        state = torch.load(Path(ckpt).expanduser(), map_location="cpu", weights_only=False)
        self.net.load_state_dict(state.get("model_state_dict", state), strict=True)

    @property
    def compute_dtype(self) -> torch.dtype:
        return next(self.net.parameters()).dtype

    def decimation(self, time_ns: jt.Int64[torch.Tensor, "L"]) -> int:
        """Native samples per net sample, ``round(rate / 40)``, from the median sample interval (some
        caches carry duplicate or backward timestamps)."""
        dt_ns = time_ns.diff().double()
        dt_ns = dt_ns[dt_ns > 0]
        assert dt_ns.numel() > 0, "stream has no strictly increasing timestamps"
        rate = 1e9 / float(dt_ns.median())
        return max(1, int(round(rate / self.net_hz)))

    def body_features(
        self,
        batch   : IMUData_w_GT_Jagged,
        row_keep: jt.Bool[torch.Tensor, "pack"],
        run_keep: jt.Bool[torch.Tensor, "N"],
        T       : int,
    ) -> jt.Float[torch.Tensor, "T 6"]:
        """Per-row body-frame ``[gyro; acc]`` (official channel order) for the kept run, fp32, acc
        completed to specific force as in ``AirIO._body_features``."""
        R_wb = pp.SO3(batch.gt_pose.rotation().tensor()[row_keep])
        g_up = torch.tensor([0.0, 0.0, GRAVITY_STANDARD], device=batch.acc.device)
        g_w  = (batch.gravity.float()[run_keep] + g_up).repeat_interleave(T, dim=0)
        acc  = batch.acc.float()[row_keep] + tp.cast(torch.Tensor, R_wb.Inv() @ g_w).float()
        return torch.cat([batch.gyro.float()[row_keep], acc], dim=-1)

    def _encode(self, windows: jt.Float[torch.Tensor, "n 6 NET_WINDOW"]) -> jt.Float[torch.Tensor, "n C"]:
        """The trunk's ResNet stack, i.e. ``ResNetLSTMSeqNet.forward`` up to the LSTM."""
        trunk = _attr(self.net, "model")
        x = trunk.input_block(windows)
        x = trunk.residual_groups(x)
        return trunk.resnet_post_pro(x).flatten(1)

    @torch.inference_mode()
    def stream(
        self,
        feat : jt.Float[torch.Tensor, "L 6"],
        step : int,
        chunk: int = 4096,
    ) -> TartanIMU_Output:
        """The official dense evaluation over one stream, encoding each stride-1 decimated window once.

        Every window start is a multiple of ``step`` and the LSTM is zero-initialised per sequence, so
        this matches the vendored sequence loop at a tenth of the cost. Its ``max_v_norm`` filter and
        ``clip_velocity_outliers`` are deliberately not applied. Streams shorter than one
        ``SEQ_LEN``-window sequence return an empty output.
        """
        trunk  = _attr(self.net, "model")
        head   = _attr(self.net, "heads")[self.head]
        device = feat.device

        y     = feat[::step].to(self.compute_dtype)              # [M 6], the ~40 Hz stream
        n_win = y.shape[0] - NET_WINDOW + 1                      # windows at every y offset
        n_seq = n_win - (SEQ_LEN - 1) * NET_WINDOW               # sequences with a full warm-up
        if n_seq <= 0:
            empty = torch.zeros(0, 3, device=device, dtype=self.compute_dtype)
            return TartanIMU_Output(vel=empty, centers=torch.zeros(0, dtype=torch.int64, device=device))

        # [n_win, 6, NET_WINDOW] view: window w spans y[w : w + NET_WINDOW].
        windows = y.unfold(0, NET_WINDOW, 1)
        embed   = torch.cat([
            self._encode(windows[c0 : c0 + chunk].contiguous()) for c0 in range(0, n_win, chunk)
        ])

        trunk.lstm.flatten_parameters()
        base = torch.arange(n_seq, device=device)
        off  = torch.arange(SEQ_LEN, device=device) * NET_WINDOW
        vels = []
        for c0 in range(0, n_seq, chunk):
            seq_embed = embed[base[c0 : c0 + chunk, None] + off[None, :]]   # [b SEQ_LEN C]
            out, _    = trunk.lstm(seq_embed)                               # zero-initialised state
            last      = out[:, -1].contiguous()
            # Heads are position-independent MLPs, so the last step alone matches ``pred[:, -1]``.
            # The head emits body velocity in m/s directly.
            vels.append(head(last, last.shape[0], 1).reshape(-1, 3))

        # Prediction s is read out from window ``s + 9 * NET_WINDOW`` and anchored at that window's centre.
        centers = (base + (SEQ_LEN - 1) * NET_WINDOW) * step + (NET_WINDOW * step) // 2
        return TartanIMU_Output(vel=torch.cat(vels), centers=centers)

    def forward(self, batch: IMUData_w_GT_Jagged) -> TartanIMU_Output:
        """Single-run convenience wrapper; the evaluator drives ``stream`` per device run."""
        run_len = batch.offsets_N.diff()
        assert run_len.numel() == 1, f"forward takes one device run, got {run_len.numel()}"
        T        = int(run_len[0])
        row_keep = torch.ones(T, dtype=torch.bool, device=batch.acc.device)
        run_keep = torch.ones(1, dtype=torch.bool, device=batch.acc.device)
        feat     = self.body_features(batch, row_keep, run_keep, T)
        return self.stream(feat, self.decimation(batch.time_ns[:, 0]))

    def min_stream_len(self, step: int) -> int:
        """Shortest native stream that yields one prediction: ``SEQ_LEN * NET_WINDOW`` decimated samples."""
        return (SEQ_LEN * NET_WINDOW - 1) * step + 1
