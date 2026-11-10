"""Import-time runtime setup for the ``src`` package; runs before any submodule or Hydra composition."""

import os
import torch
import argparse
import warnings
from pathlib import Path

from omegaconf import OmegaConf

# Lightning still constructs torch's deprecated LeafSpec; silence just that warning.
warnings.filterwarnings(
    "ignore",
    message=r"`isinstance\(treespec, LeafSpec\)` is deprecated",
    category=FutureWarning,
)

torch.set_float32_matmul_precision("high")
# DDP keeps AccumulateGrad nodes across iterations, tripping this warning on every backward.
if hasattr(torch.autograd.graph, "set_warn_on_accumulate_grad_stream_mismatch"):
    torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)

# Python 3.14's argparse _check_help rejects Hydra 1.3.2's LazyCompletionHelp; skip the validator.
if hasattr(argparse.ArgumentParser, "_check_help"):
    argparse.ArgumentParser._check_help = lambda *_: None  # type: ignore[attr-defined]


# Use in yaml as ${machine-conf:data_root}/nymeria_imu/... or ${machine-conf:exp_root,./experiments}.

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ROOT_DEFAULTS = {"data_root": "data", "cache_root": "data", "exp_root": "experiments"}

def machine_conf(key: str, default: str | None = None) -> str:
    """Resolve a root from FOUNDATIONIMU_{DATA,CACHE,EXP}_ROOT; relative values anchor to the repo."""
    if key not in _ROOT_DEFAULTS:
        if default is None:
            raise ValueError(f"machine-conf: unknown root {key!r}")
        value = default
    else:
        value = os.environ.get(f"FOUNDATIONIMU_{key.upper()}") or default or _ROOT_DEFAULTS[key]
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else _REPO_ROOT / path)


OmegaConf.register_new_resolver("machine-conf", machine_conf, replace=True)
# ${len:${model.feature_from_trunk}} ties tokenizer.num_trunk_features to the trunk layers consumed.
OmegaConf.register_new_resolver("len", lambda seq: len(seq), replace=True)
