"""Adapter around the frozen ImageBind IMU and text towers vendored at ``thirdparty/imagebind``:
single-site IMU embeddings in ImageBind's 1024-d joint space, evaluated as released."""

import sys
import types
import importlib
import itertools

from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import jaxtyping as jt

import typing as tp

from ..interface.data_imu import IMUData_BaseTrait_Jagged
from ..interface.human import CANONICAL_IDX, T_CANONICAL_NAMES

_IMAGEBIND_ROOT = Path(__file__).resolve().parents[2] / "thirdparty/imagebind"
_BPE_PATH       = _IMAGEBIND_ROOT / "imagebind/bpe/bpe_simple_vocab_16e6.txt.gz"

# Pinned by ImageBind's learned position embedding; Ego4D IMU is 200 Hz.
IMU_SAMPLES  = 2000
IMU_RATE_HZ  = 200.0
WINDOW_MS    = IMU_SAMPLES / IMU_RATE_HZ * 1e3          # ms of real time per input

# The site closest to the Ego4D head-mounted IMU ImageBind was trained on.
DEFAULT_SENSOR: T_CANONICAL_NAMES = "head"

# This repo's body frame is the Xsens MVN segment convention: X forward, Y left, Z up. ImageBind's own
# convention is undocumented, so ``frame`` selects one (see ``FRAME_SPECS``).
DEFAULT_FRAME = "+x+y+z"

_AXES = {"x": 0, "y": 1, "z": 2}

# ImageBind keeps one entry per modality in each of these ``nn.ModuleDict``s.
_MODALITY_GROUPS = (
    "modality_preprocessors",
    "modality_trunks",
    "modality_heads",
    "modality_postprocessors",
)

# Width of ImageBind's joint space.
OUT_EMBED_DIM = 1024

# ``imagebind_huge`` hyper-parameters; its own builder hard-codes a download path and builds every branch.
_HUGE_KWARGS = dict(
    vision_embed_dim = 1280, vision_num_blocks = 32, vision_num_heads = 16,
    text_embed_dim   = 1024, text_num_blocks   = 24, text_num_heads   = 16,
    out_embed_dim    = OUT_EMBED_DIM,
    audio_drop_path  = 0.1,
    imu_drop_path    = 0.7,
)


def frame_matrix(spec: str) -> torch.Tensor:
    """The 3x3 rotation a ``"+x+y+z"``-style spec denotes, applied as ``v_target = R @ v_ours``.

    ``spec`` names which of our axes each target axis is: ``"-z+x+y"`` means the encoder's X is our -Z,
    its Y our +X, its Z our +Y. Only proper rotations are accepted: under a reflection the gyroscope (a
    pseudovector) would need an extra sign, so the two channels could not share one matrix.
    """
    assert len(spec) == 6, f"frame spec must be 6 chars like '+x-z+y', got {spec!r}"
    R = torch.zeros(3, 3)
    seen: set[int] = set()
    for row, (sign, axis) in enumerate(zip(spec[0::2], spec[1::2])):
        assert sign in "+-" and axis in _AXES, f"bad term {sign}{axis!r} in frame spec {spec!r}"
        assert _AXES[axis] not in seen, f"frame spec {spec!r} uses axis {axis!r} twice"
        seen.add(_AXES[axis])
        R[row, _AXES[axis]] = 1.0 if sign == "+" else -1.0
    det = float(torch.det(R))
    assert abs(det - 1.0) < 1e-6, (
        f"frame spec {spec!r} has determinant {det:+.0f}; it is a reflection, not a rotation "
        f"(a mirrored frame would flip the gyroscope's sign but not the accelerometer's)"
    )
    return R


def all_frames() -> list[str]:
    """Every right-handed signed axis permutation: the 24 orientations a rigid sensor can be mounted in."""
    specs = []
    for axes in itertools.permutations("xyz"):
        for signs in itertools.product("+-", repeat=3):
            spec = "".join(s + a for s, a in zip(signs, axes))
            if abs(float(torch.det(frame_matrix_unchecked(spec))) - 1.0) < 1e-6:
                specs.append(spec)
    return specs


def frame_matrix_unchecked(spec: str) -> torch.Tensor:
    """``frame_matrix`` without the determinant check, so ``all_frames`` can filter on it."""
    R = torch.zeros(3, 3)
    for row, (sign, axis) in enumerate(zip(spec[0::2], spec[1::2])):
        R[row, _AXES[axis]] = 1.0 if sign == "+" else -1.0
    return R


FRAME_SPECS: dict[str, str] = {
    # Our own frame, the default.
    "xsens"  : "+x+y+z",
    # Camera convention (X right, Y down, Z forward): our +X -> +Z, +Y -> -X, +Z -> -Y.
    "camera" : "-y-z+x",
}


def _imagebind_module(name: str) -> types.ModuleType:
    """Import ``imagebind.<name>`` from the vendored checkout, skipping its ``__init__``.

    The modules use absolute ``imagebind.*`` imports, so the package must be importable, but its
    ``__init__.py`` pulls in ``torchaudio`` and ``pytorchvideo``. A bare module whose ``__path__``
    points at the checkout avoids both.
    """
    if "imagebind" not in sys.modules:
        assert _BPE_PATH.exists(), (
            f"ImageBind checkout not found at {_IMAGEBIND_ROOT}; clone "
            f"https://github.com/facebookresearch/ImageBind into thirdparty/imagebind"
        )
        pkg = types.ModuleType("imagebind")
        pkg.__path__ = [str(_IMAGEBIND_ROOT / "imagebind")]   # type: ignore[attr-defined]
        sys.modules["imagebind"] = pkg
        sys.path.insert(0, str(_IMAGEBIND_ROOT))              # for `imagebind.*`'s own imports
    return importlib.import_module(f"imagebind.{name}")


@dataclass
class ImageBind_Output:
    embed    : jt.Float[torch.Tensor, "B 1024"]   # per-window IMU embedding (zero where ~valid)
    valid    : jt.Bool[torch.Tensor , "B"]        # False where no run of the window spans the declared window
    win_count: jt.Int64[torch.Tensor, "B"]        # runs pooled into each window


class ImageBind_IMU(nn.Module):
    """The frozen ImageBind IMU (and optionally text) tower, driven by this repo's jagged batches.

    ``modalities`` selects the branches to keep; all others are deleted before the checkpoint is read.
    """

    def __init__(
        self,
        ckpt      : str,
        sensor    : T_CANONICAL_NAMES = DEFAULT_SENSOR,
        frame     : str               = DEFAULT_FRAME,   # our axes -> the encoder's, see frame_matrix
        modalities: tp.Sequence[str]  = ("imu",),
        chunk     : int               = 256,          # max runs encoded per forward pass
        window_ms : float             = WINDOW_MS,    # real duration the caller promises to deliver
        window_tol: float             = 0.05,         # tolerated |run - window_ms| / window_ms
    ):
        super().__init__()
        assert set(modalities) <= {"imu", "text"}, f"unsupported ImageBind modalities: {modalities}"
        self.sensor: T_CANONICAL_NAMES = sensor
        self.sensor_id  = CANONICAL_IDX[sensor]
        self.frame      = FRAME_SPECS.get(frame, frame)
        self.modalities = tuple(modalities)
        # Buffer so it follows .to(device)/.to(dtype) with the encoder.
        self.register_buffer("frame_R", frame_matrix(self.frame), persistent=False)
        self.chunk      = int(chunk)
        self.window_ms  = float(window_ms)
        self.window_tol = float(window_tol)

        model_mod = _imagebind_module("models.imagebind_model")
        self.MODALITY = model_mod.ModalityType

        # Drop unused branches BEFORE loading, so only the kept tensors are read off the checkpoint.
        model = model_mod.ImageBindModel(**_HUGE_KWARGS)
        for group in _MODALITY_GROUPS:
            branches = getattr(model, group)
            for key in list(branches.keys()):
                if key not in self.modalities:
                    del branches[key]

        prefixes = tuple(f"{g}.{m}." for g in _MODALITY_GROUPS for m in self.modalities)
        path     = Path(ckpt).expanduser()
        assert path.exists(), (
            f"ImageBind checkpoint not found at {path}; download it with\n"
            f"  curl -fL --create-dirs -o {path} "
            f"https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth"
        )
        # mmap: the file is 4.5 GB and only part of it is read.
        raw   = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        state = {k: v for k, v in raw.items() if k.startswith(prefixes)}
        missing, _ = model.load_state_dict(state, strict=False)
        assert not missing, f"{path} is missing ImageBind weights: {missing[:8]}"

        self.model: nn.Module = model.eval().requires_grad_(False)
        self._tokenizer = None      # built on first encode_text

    @property
    def compute_dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    @property
    def warp(self) -> float:
        """Time stretch of the declared window against ImageBind's training rate: 1.0 is native, ``> 1``
        means the encoder sees slow motion, ``< 1`` fast motion."""
        return WINDOW_MS / self.window_ms

    def configure_trainable_params(self) -> None:
        """Nothing trains; parity with ``FoundationIMU``."""
        self.model.requires_grad_(False)

    def encode_text(self, texts: tp.Sequence[str], batch: int = 64) -> jt.Float[torch.Tensor, "K 1024"]:
        """Embed ``texts`` with ImageBind's text tower (needs ``"text"`` in ``modalities``).

        Rows are the model's own output (unit-norm times a learned logit scale); callers re-normalize.
        """
        assert "text" in self.modalities, "ImageBind_IMU was built without the text tower"
        if self._tokenizer is None:
            prep = _imagebind_module("models.multimodal_preprocessors")
            self._tokenizer = prep.SimpleTokenizer(bpe_path=str(_BPE_PATH))

        device = next(self.model.parameters()).device
        out: list[torch.Tensor] = []
        for i in range(0, len(texts), batch):
            tokens = torch.stack([self._tokenizer(t) for t in texts[i : i + batch]]).to(device)
            with torch.inference_mode():
                out.append(self.model({self.MODALITY.TEXT: tokens})[self.MODALITY.TEXT].float().cpu())
        return torch.cat(out, dim=0) if out else torch.empty(0, OUT_EMBED_DIM)

    def _resample_runs(
        self,
        imu6   : jt.Float[torch.Tensor, "pack 6"],
        starts : jt.Int64[torch.Tensor, "N"],
        lengths: jt.Int64[torch.Tensor, "N"],
    ) -> jt.Float[torch.Tensor, "N 6 2000"]:
        """Linearly resample every selected run to ``IMU_SAMPLES``, batched by source length."""
        n, L = lengths.numel(), IMU_SAMPLES
        out  = imu6.new_empty(n, 6, L)
        for src in torch.unique(lengths).tolist():
            sel = (lengths == src).nonzero(as_tuple=False).flatten()
            idx = starts[sel][:, None] + torch.arange(src, device=imu6.device)  # [k, src] into the pack
            seg = imu6[idx].transpose(1, 2)                                     # [k, 6, src]
            out[sel] = seg if src == L else F.interpolate(seg, size=L, mode="linear", align_corners=False)
        return out

    def _representable(
        self,
        time_ns: torch.Tensor,
        starts : torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Which runs really span ``self.window_ms``; every run is resampled to the same 2000 samples, so
        a mismatched run would be silently rescaled in frequency.

        Too long is operator error and asserts. Too short is data (a take shorter than one window is
        emitted whole) and is excluded through ``ImageBind_Output.valid``. Runs are checked against the
        declared window, not the native one (see ``warp``).
        """
        span_ms = (time_ns[starts + lengths - 1] - time_ns[starts]).float() * 1e-6
        lo, hi  = (1 - self.window_tol) * self.window_ms, (1 + self.window_tol) * self.window_ms
        assert bool((span_ms <= hi).all()), (
            f"this ImageBind_IMU was built for {self.window_ms:.0f} ms windows, but the batch "
            f"carries runs up to {float(span_ms.max()):.0f} ms; the encoder resamples every run to "
            f"{IMU_SAMPLES} samples, so a longer run would be compressed in time. Rebuild it with "
            f"window_ms matching the pipeline (native is {WINDOW_MS:.0f})."
        )
        return span_ms >= lo

    def forward(
        self,
        batch : IMUData_BaseTrait_Jagged,
        sensor: T_CANONICAL_NAMES | None = None,
        frame : str | None               = None,
    ) -> ImageBind_Output:
        """Embed each window's ``sensor`` run in axis convention ``frame`` (default: the instance's).

        The overrides let placement and frame sweeps reuse one data pass and one loaded model.
        """
        assert batch.kind != "filtered-no-g", (
            "ImageBind's IMU tower was trained on head-camera accelerometer readings, which carry "
            "gravity; this batch's `acc` is gravity-free. Use a pipeline whose transform stack "
            "adds gravity back (the suite's add_imu_noise does)."
        )
        site      = self.sensor if sensor is None else sensor
        site_id   = CANONICAL_IDX[site]
        device    = batch.acc.device
        offsets_N = batch.offsets_N
        offsets_B = batch.offsets_B
        starts    = offsets_N[:-1]
        lengths   = offsets_N.diff()
        B         = offsets_B.numel() - 1

        # sensor_id is constant within a run, so read it at the run start.
        keep = batch.sensor_id[starts, 0] == site_id
        assert bool(keep.any()), (
            f"no window in this batch carries a {site} run; this baseline is single-IMU and "
            f"needs a {site} IMU (canonical id {site_id})"
        )
        starts, lengths = starts[keep], lengths[keep]
        # ... and, of those, only the runs that span the declared window.
        long_enough = self._representable(batch.time_ns[:, 0], starts, lengths)
        starts, lengths = starts[long_enough], lengths[long_enough]
        win_of_run = torch.searchsorted(offsets_B, starts, right=True) - 1
        count = torch.zeros(B, device=device, dtype=torch.int64).index_add_(
            0, win_of_run, torch.ones_like(win_of_run)
        )
        valid = count > 0

        # acc and gyro share one matrix because it is a proper rotation (see ``frame_matrix``).
        # cast past nn.Module.__getattr__'s ``Tensor | Module`` widening on the registered buffer.
        R    = tp.cast(torch.Tensor,
                       self.frame_R if frame is None else frame_matrix(FRAME_SPECS.get(frame, frame)))
        R    = R.to(device=device, dtype=torch.float32)
        acc  = batch.acc.float()  @ R.T
        gyro = batch.gyro.float() @ R.T
        imu6 = torch.cat([acc, gyro], dim=-1)                                 # Ego4D order
        x    = self._resample_runs(imu6, starts, lengths)                     # [N, 6, 2000]

        param_device = next(self.model.parameters()).device
        embs: list[torch.Tensor] = []
        for i in range(0, x.shape[0], self.chunk):
            with torch.inference_mode():
                seg = self.model({
                    self.MODALITY.IMU: x[i : i + self.chunk].to(param_device, self.compute_dtype)
                })[self.MODALITY.IMU]
            embs.append(seg.float().to(device))
        width   = OUT_EMBED_DIM
        run_emb = torch.cat(embs, dim=0) if embs else x.new_zeros(0, width)

        # Mean over the rare multi-run case; windows with no representable run keep the zero row.
        emb_sum = run_emb.new_zeros(B, width).index_add_(0, win_of_run, run_emb)
        embed   = emb_sum / count.clamp(min=1)[:, None].to(emb_sum.dtype)
        return ImageBind_Output(embed=embed, valid=valid, win_count=count)
