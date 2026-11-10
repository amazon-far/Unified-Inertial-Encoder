"""Compare FoundationIMU_TRT (src/model/model_trt.py) with the jagged model; needs a GPU and the checkpoint."""
import os
from pathlib import Path

import pytest
import torch

from src.eval import inference, load_model
from src.interface.data_imu import IMUData_Jagged
from src.model.model_trt import FoundationIMU_TRT
from src.model.template import jagged as jag_module
from src.scripts.effiency_profiling.__main__ import PAPER_CKPT, make_stream

N, RATE, DUR = 12, 200.0, 4.0
T = int(RATE * DUR)
CUDA = torch.device("cuda")
pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and PAPER_CKPT.exists()), reason="needs a GPU and the paper checkpoint")


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def jagged_to_dense(batch: IMUData_Jagged):
    """One-sample jagged pack (device-major) -> padded [N, T] arrays + per-device sensor ids."""
    offn = batch.offsets_N
    lens = (offn[1:] - offn[:-1]).tolist()
    acc = torch.zeros(N, T, 3); gyro = torch.zeros(N, T, 3); tns = torch.zeros(N, T, dtype=torch.int64); valid = torch.zeros(N, T, dtype=torch.bool)
    sensor = torch.zeros(N, dtype=torch.int64)
    for n, (lo, ln) in enumerate(zip(offn[:-1].tolist(), lens)):
        acc[n, :ln] = batch.acc[lo:lo + ln].float(); gyro[n, :ln] = batch.gyro[lo:lo + ln].float()
        tns[n, :ln] = batch.time_ns[lo:lo + ln, 0]; valid[n, :ln] = True
        sensor[n] = int(batch.sensor_id[lo, 0])
    return acc, gyro, tns, valid, sensor, lens


def dense_to_pack(x: torch.Tensor, lens: list[int]) -> torch.Tensor:
    return torch.cat([x[n, :ln] for n, ln in enumerate(lens)], dim=0)


def irregular_stream(seed: int = 1) -> IMUData_Jagged:
    """Drop random samples per device (different counts) so devices differ in length and token layout."""
    full = make_stream(N, RATE, DUR, seed=seed)
    gen = torch.Generator().manual_seed(seed)
    keep_rows = []
    for n in range(N):
        n_keep = int(T * (0.55 + 0.45 * torch.rand((), generator=gen)))
        idx = torch.randperm(T, generator=gen)[:n_keep].sort().values + n * T
        keep_rows.append(idx)
    lens = torch.tensor([len(r) for r in keep_rows])
    rows = torch.cat(keep_rows)
    pick = lambda t: t[rows]
    return IMUData_Jagged(
        time_ns=pick(full.time_ns), acc=pick(full.acc), gyro=pick(full.gyro), correct_mask=pick(full.correct_mask),
        correct_acc=pick(full.correct_acc), correct_gyro=pick(full.correct_gyro), gravity=full.gravity, correct_gravity=full.correct_gravity,
        acc_bias=pick(full.acc_bias), gyro_bias=pick(full.gyro_bias),
        offsets_B=torch.tensor([0, int(lens.sum())]), offsets_N=torch.cat([torch.zeros(1, dtype=torch.int64), lens.cumsum(0)]),
        sensor_id=pick(full.sensor_id), model=full.model, kind=full.kind,
    )


@pytest.fixture(scope="module")
def model():
    os.environ["FIMU_EVAL_WEIGHTS"] = "ema"
    return load_model(PAPER_CKPT).eval().cuda()


@pytest.fixture(scope="module")
def trt(model):
    return FoundationIMU_TRT(max_samples=T, max_timespan_ms=DUR * 1e3, num_devices=N, precision="fp16").from_torch_model(model)


def compare(trt, model, batch, backend, tol):
    ref = inference(model, batch)
    acc, gyro, tns, valid, sensor, lens = jagged_to_dense(batch)
    out = trt(acc, gyro, tns, valid, sensor, backend=backend)
    errs = {
        "correct_acc":  rel(dense_to_pack(out.dec_correct_acc, lens),  ref.dec_correct_acc),
        "correct_gyro": rel(dense_to_pack(out.dec_correct_gyro, lens), ref.dec_correct_gyro),
        "body_vel":     rel(dense_to_pack(out.dec_body_vel, lens),     ref.dec_body_vel),
        "lang":         rel(out.dec_lang_feature,                      ref.dec_lang_feature[0, 0]),
    }
    n_tok_ref = int(ref.offsets_N_feat[-1]) - 1 - model.tokenizer.num_register
    n_tok = int(out.token_valid.sum())
    print(f"\n[{backend}] tokens ref={n_tok_ref} dense={n_tok}  rel-L2 errors: " + ", ".join(f"{k}={v:.2e}" for k, v in errs.items()))
    assert n_tok == n_tok_ref
    for k, v in errs.items():
        assert v < tol, f"{k}: rel err {v:.3e} >= {tol}"


def test_trt_regular_stream(trt, model):
    compare(trt, model, make_stream(N, RATE, DUR).to(CUDA, torch.bfloat16), "trt", tol=5e-2)


def test_trt_irregular_stream(trt, model):
    compare(trt, model, irregular_stream().to(CUDA, torch.bfloat16), "trt", tol=5e-2)


def test_torch_backend_regular_stream(trt, model):
    compare(trt, model, make_stream(N, RATE, DUR).to(CUDA, torch.bfloat16), "torch", tol=5e-2)


def test_dense_matches_fp32_jagged(trt):
    """Tight check of the dense re-formulation: both sides in fp32 (jagged via the SDPA fallback attention)."""
    ref_model = load_model(PAPER_CKPT).eval().float().cuda()
    for blk in ref_model.trunk.blocks:
        setattr(blk.attn, "__class__", jag_module.Attention_on_OtherPlatforms)   # dense-mask SDPA instead of flash varlen
    for batch in (make_stream(N, RATE, DUR), irregular_stream(3)):
        compare(trt, ref_model, batch.to(CUDA, torch.float32), "torch", tol=2e-3)


def test_schedule_reuse_and_padding(trt, model):
    """A short stream (fewer samples, shorter span) padded into the fixed grid still matches."""
    short = make_stream(N, RATE, 2.0).to(CUDA, torch.bfloat16)      # 400 samples/device inside an 800-slot grid
    ref = inference(model, short)
    acc, gyro, tns, valid, sensor, lens = jagged_to_dense(short)
    sched = trt.schedule(tns, valid, sensor)
    out1 = trt(acc, gyro, schedule=sched)
    out2 = trt(acc, gyro, schedule=sched)
    assert torch.equal(out1.dec_body_vel, out2.dec_body_vel)
    assert rel(dense_to_pack(out1.dec_body_vel, lens), ref.dec_body_vel) < 5e-2
    assert not out1.dec_body_vel[:, 400:].any()   # padded rows are zeroed


def test_rejects_non_ours_module(model):
    with pytest.raises(ValueError):
        FoundationIMU_TRT(max_samples=8, max_timespan_ms=250., num_devices=1).from_torch_model(model.trunk)  # type: ignore[arg-type]


def test_error_budget_against_fp32_reference(trt, model):
    """Compare the fp16 engine and the bf16 checkpoint with the fp32 jagged model on the irregular stream."""
    ref32 = load_model(PAPER_CKPT).eval().float().cuda()
    for blk in ref32.trunk.blocks:
        setattr(blk.attn, "__class__", jag_module.Attention_on_OtherPlatforms)
    batch = irregular_stream(5)
    r32 = inference(ref32, batch.to(CUDA, torch.float32))
    r16 = inference(model, batch.to(CUDA, torch.bfloat16))
    acc, gyro, tns, valid, sensor, lens = jagged_to_dense(batch)
    out = trt(acc, gyro, tns, valid, sensor)
    rows = []
    for name, ours, bf16, truth in (
        ("correct_acc", dense_to_pack(out.dec_correct_acc, lens), r16.dec_correct_acc, r32.dec_correct_acc),
        ("body_vel",    dense_to_pack(out.dec_body_vel, lens),    r16.dec_body_vel,    r32.dec_body_vel),
        ("lang",        out.dec_lang_feature,                     r16.dec_lang_feature[0, 0], r32.dec_lang_feature[0, 0]),
    ):
        e_trt, e_bf16 = rel(ours, truth), rel(bf16, truth)
        rows.append((name, e_trt, e_bf16))
        assert e_trt < 3e-2 and e_trt < 2 * max(e_bf16, 5e-3), f"{name}: fp16 engine {e_trt:.2e} vs bf16 checkpoint {e_bf16:.2e} (both vs fp32)"
    print("\nrel-L2 vs fp32 reference: " + ", ".join(f"{n}: trt-fp16={a:.2e} bf16-ckpt={b:.2e}" for n, a, b in rows))
