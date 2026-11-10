"""Verify ``src.model.tartanimu.TartanIMU.stream`` against the official TartanIMU test loop."""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model.tartanimu import NET_WINDOW, SEQ_LEN, TartanIMU, _attr  # noqa: E402
from src.runtime_prologue import machine_conf  # noqa: E402

WEIGHTS = Path(machine_conf("cache_root")) / "tartanimu"
HEAD    = "human"


def _model() -> TartanIMU:
    if not (WEIGHTS / "unified.pt").exists():
        pytest.skip(f"released TartanIMU weights not at {WEIGHTS}; see src/config/model/tartanimu.yaml")
    model = TartanIMU(
        ckpt         = str(WEIGHTS / "unified.pt"),
        config       = str(WEIGHTS / "config" / "unified.yaml"),
        model_config = str(WEIGHTS / "config" / "resnet_lstm_multihead.yaml"),
        head         = HEAD,
    )
    return model.eval()


@torch.no_grad()
def _official(model: TartanIMU, feat: torch.Tensor, step: int) -> torch.Tensor:
    """The vendored path written out literally: official slicing, one forward, ``pred[:, -1]``."""
    W = NET_WINDOW * step                       # native samples per 1 s window
    L = feat.shape[0]
    starts = range(0, L - SEQ_LEN * W + 1, step)   # BasicSequenceData's index map stride
    seqs = []
    for j in starts:
        windows = [feat[j + i * W : j + (i + 1) * W][::step].T for i in range(SEQ_LEN)]
        seqs.append(torch.stack(windows))       # [10, 6, 40]
    x = torch.stack(seqs)                       # [n_seq, 10, 6, 40]

    label = torch.full((x.shape[0],), 4, dtype=torch.long)   # MOTION_TYPE_ID["human"]
    out = model.net(x, motion_type=label, predict_cov=False, compute_all_heads=False)
    return out[HEAD][:, -1]                     # fun_test_forward's readout


@pytest.mark.parametrize("step", [3, 5, 6])
def test_stream_matches_official_loop(step: int) -> None:
    torch.manual_seed(0)
    model = _model()

    # Body-frame [gyro; acc], long enough for a few hundred sequences at every decimation.
    L    = SEQ_LEN * NET_WINDOW * step + 137 * step
    feat = torch.randn(L, 6) * torch.tensor([0.5, 0.5, 0.5, 1.0, 1.0, 1.0])
    feat[:, 5] += 9.81                           # gravity in the body-z accelerometer channel

    ours = model.stream(feat, step)
    ref  = _official(model, feat, step)

    n = min(ours.vel.shape[0], ref.shape[0])
    assert n > 100, f"test stream produced only {n} predictions"
    torch.testing.assert_close(ours.vel[:n], ref[:n], rtol=1e-5, atol=1e-6)

    # Anchors: pose_integrate's ind_intg == index_map start + (seq_len - 1) * W + W // 2.
    W        = NET_WINDOW * step
    expected = torch.arange(n) * step + (SEQ_LEN - 1) * W + W // 2
    torch.testing.assert_close(ours.centers[:n], expected)


def test_stream_is_empty_below_ten_windows() -> None:
    """Strict official warm-up: a stream shorter than ten windows yields nothing at all."""
    model = _model()
    step  = 6
    short = torch.randn(model.min_stream_len(step) - 1, 6)
    assert model.stream(short, step).vel.shape[0] == 0
    assert model.stream(torch.randn(model.min_stream_len(step), 6), step).vel.shape[0] == 1


def test_head_readout_matches_full_sequence_readout() -> None:
    """The heads are position-wise, so a last-step readout equals slicing a full-sequence readout."""
    model = _model()
    head  = _attr(model.net, "heads")[HEAD]

    torch.manual_seed(1)
    lstm_out = torch.randn(7, SEQ_LEN, model.cfg["model_param"]["lstm_size"])
    with torch.no_grad():
        full = head(lstm_out.reshape(-1, lstm_out.shape[-1]), 7, SEQ_LEN)[:, -1]
        last = head(lstm_out[:, -1].contiguous(), 7, 1).reshape(-1, 3)
    torch.testing.assert_close(full, last)
