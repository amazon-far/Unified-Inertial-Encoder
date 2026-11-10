"""Evaluate a FoundationIMU checkpoint on one sequence; also the shared model and data loaders.

Usage:
    python -m src.eval --ckpt <run>/checkpoints/<name>.ckpt [--data <data_config>] [--subset valid]
"""

import os
import functools
import hydra
import torch
import typing as tp
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from . import evaluate
from .model.trainer_util import precision_dtypes
from .data.util import corpus_pipelines
from .interface.sequence import ISequence
from .interface.data_imu import IMUData_BaseTrait_Jagged, IMUData_LabelTrait_Jagged
from .model.model import FoundationIMU
from .utility.lazy import LazyMap
from .interface.model import FoundationIMU_Output


def load_model(ckpt: Path) -> FoundationIMU:
    """Recover a model from its run's ``.hydra`` config and the checkpoint."""
    ckpt    = Path(ckpt).expanduser()
    run_dir = ckpt.parent.parent    # checkpoints live at <run_dir>/checkpoints/<ckpt>
    cfg     = OmegaConf.load(run_dir / ".hydra" / "config.yaml")

    model: FoundationIMU = hydra.utils.instantiate(cfg.model)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    # EMA when present; FIMU_EVAL_WEIGHTS=online forces the optimizer weights, needed for
    # cooldown-trained runs, whose bf16 EMA shadow stalls at small learning rates.
    if "ema_state_dict" in state and os.environ.get("FIMU_EVAL_WEIGHTS", "ema") != "online":
        model.load_state_dict(state["ema_state_dict"])
    else:
        model.load_state_dict({k.removeprefix("model."): v for k, v in state["state_dict"].items() if k.startswith("model.")})

    # Match the run's training precision; the flash-attn trunk rejects fp32.
    return model.to(precision_dtypes(cfg.trainer.precision)[0])


def load_sequence(
    data_cfg : Path | None = None,
    subset   : tp.Literal["train", "valid"]="valid",
    overrides: tp.Sequence[str] = (),
    ckpt     : Path | None = None
) -> LazyMap[DictConfig, ISequence[IMUData_BaseTrait_Jagged]]:
    """Lazily load one subset of a ``config/data`` yaml, or of the ckpt run's own config if ``data_cfg`` is None."""
    if data_cfg is None:
        assert ckpt is not None, "load_sequence needs a data_cfg, a ckpt (to reuse its run config), or both"
        assert not overrides, "overrides require recomposition; pass data_cfg alongside them"
        cfg = OmegaConf.load(Path(ckpt).expanduser().parent.parent / ".hydra" / "config.yaml")
        return _expand_corpora(cfg.data[subset])

    data_choice = Path(data_cfg).stem
    # Data configs interpolate root keys, so they must be composed under the train root.
    with hydra.initialize_config_dir(config_dir=str(Path(__file__).resolve().parent / "config"), version_base=None):
        if ckpt is not None:
            overrides = [*_transform_overrides_from_ckpt(ckpt, data_choice, overrides), *overrides]
        cfg = hydra.compose(config_name="train", overrides=[f"data={data_choice}", *overrides])
    return _expand_corpora(cfg.data[subset])


def _expand_corpora(entries) -> LazyMap:
    """One LazyMap item per recording; ``build_corpus`` entries expand to one builder per stem."""
    xs: list = []
    for spec in entries:
        if str(spec.get("_target_", "")).endswith("build_corpus"):
            container = OmegaConf.to_container(spec, resolve=True)
            assert isinstance(container, dict)
            kwargs = {str(k): v for k, v in container.items() if not str(k).startswith("_")}
            try:
                stems, build_one = corpus_pipelines(**kwargs)
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"{e}\nThe stems listing recorded in this run's config no longer exists in "
                    f"the repo (renamed/removed since the run). Pass data_cfg (CLI: --data "
                    f"<config>) to recompose against the current listings instead."
                ) from e
            xs.extend(functools.partial(build_one, stem) for stem in stems)
        else:
            xs.append(spec)
    return LazyMap(lambda x: x() if callable(x) else hydra.utils.instantiate(x), xs)


def _transform_overrides_from_ckpt(ckpt: Path, data_choice: str, user_overrides: tp.Sequence[str]) -> list[str]:
    """Derive ``data/transform*`` overrides matching the checkpoint's training preprocess stack.

    Must be called inside an active ``hydra.initialize*`` context. A bare eval slot falls back to
    the training slot whose source name (``data.nymeria_t`` -> ``nymeria``) prefixes ``data_choice``.
    """
    run_choices = OmegaConf.load(Path(ckpt).expanduser().parent.parent / ".hydra" / "hydra.yaml").hydra.runtime.choices
    run_tf      = {k: v for k, v in run_choices.items() if k.startswith("data/transform")}

    probe     = hydra.compose(config_name="train", overrides=[f"data={data_choice}"], return_hydra_config=True)
    eval_keys = [k for k in probe.hydra.runtime.choices if k.startswith("data/transform")]
    user_keys = {o.split("=", 1)[0].lstrip("+~") for o in user_overrides}

    derived = []
    for key in eval_keys:
        if key in user_keys:
            continue
        if key in run_tf:
            derived.append(f"{key}={run_tf[key]}")
        elif key == "data/transform":
            source  = lambda k: k.split("@")[-1].removeprefix("data.").removesuffix("_t")
            matched = {v for k, v in run_tf.items() if data_choice.startswith(source(k))}
            if len(matched) == 1:
                derived.append(f"{key}={matched.pop()}")
            else:
                print(f"[load_sequence] cannot derive a transform for '{key}' from the ckpt run "
                      f"(candidates: {sorted(matched) or list(run_tf.values())}); pass --override {key}=<transform> explicitly.")
    if derived:
        print(f"[load_sequence] applying training-time transforms from ckpt: {derived}")
    return derived


def inference(
    model: FoundationIMU,
    batch: IMUData_BaseTrait_Jagged
) -> FoundationIMU_Output:
    """Run the model over one pre-collated jagged batch (``seq[:]``) in a single forward.

    Takes the batch, not the sequence: window/transform layers redraw on every fetch, so the
    caller must hand this and the metrics the same batch to keep their rows aligned.
    """
    device = next(model.parameters()).device
    dtype  = next(model.parameters()).dtype

    batch = batch.to(device, dtype)
    mask  = torch.zeros(batch.acc.shape[0], dtype=torch.bool, device=device)

    with torch.inference_mode():
        return model(batch, mask)


if __name__ == "__main__":
    import argparse
    import rerun as rr
    import rerun.blueprint as rrb

    parser = argparse.ArgumentParser()
    parser.add_argument("--data"  , type=str, default=None, help="*.yaml file in ./config/data that points to the [valid] subset to run inference on.")
    parser.add_argument("--ckpt"  , type=str, help="*.pth  file in <machine_conf:exp_root> to run inference on.")
    parser.add_argument("--subset", type=str, choices=["train", "valid"], default="valid", help="which subset to use for inference.")
    parser.add_argument("--override", action="append", default=[], help="Hydra override for the data composition (e.g. data/transform=add_g_noise_rand_dropdev)")
    args   = parser.parse_args()

    model   = load_model(args.ckpt).eval().cuda()
    datas   = load_sequence(args.data, args.subset, args.override, ckpt=args.ckpt)

    # Collate once and share the batch with the metrics (see `inference`).
    idx = 0
    raw_batch    = datas[idx][:]
    infer_result = inference(model, raw_batch)
    batch        = tp.cast(IMUData_LabelTrait_Jagged, raw_batch)

    metrices = [
        evaluate.Velocity_Metric(batch, infer_result),
        evaluate.Trajectory_Metric(batch, infer_result)
    ]

    # Recording must be active before show(), which mirrors metrics as rr TextLog rows.
    rr.init("FoundationIMU_Eval", spawn=False)
    rr.save("./temp/result.rrd")
    rr.send_blueprint(rrb.Blueprint(rrb.Horizontal(contents=[
        rrb.TextLogView(),
        rrb.Tabs(contents=[x.rr_blueprint() for x in metrices]),
    ], column_shares=[1, 3]), collapse_panels=True))
    
    for metric in metrices:
        metric.show()
        metric.rr_log()
