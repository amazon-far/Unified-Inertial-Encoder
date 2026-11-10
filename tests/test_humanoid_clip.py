"""The published real-G1 clip layout loads through the cache readers."""
import json

import numpy as np
import pytest
import torch

from src.data.imu.g1_superodom_processed import Unitree_G1_SuperOdom_IMU_Processed
from src.data.imu.humanoid_superodom_npz import Humanoid_SuperOdom_NPZ_Processed_Language
from src.data.humanoid.g1_superodom import G1_WholeBody_Link_Mapping

T = 64
NAMES = list(G1_WholeBody_Link_Mapping)


@pytest.fixture
def clip(tmp_path):
    rng  = np.random.default_rng(0)
    pose = np.zeros((len(NAMES), T, 7), dtype=np.float32)
    pose[..., 6] = 1.0
    np.save(tmp_path / "imus.npy", rng.standard_normal((len(NAMES), T, 9)).astype(np.float16))
    np.save(tmp_path / "pose.npy", pose)
    np.save(tmp_path / "time_ns.npy", np.arange(T, dtype=np.int64) * 5_000_000)
    fields = {k: {"col": [3 * i, 3 * i + 3], "shape": [len(NAMES), 3]}
              for i, k in enumerate(("acc", "gyro", "gt_vel"))}
    (tmp_path / "spec.json").write_text(json.dumps({
        "format" : 3,
        "length" : T,
        "names"  : NAMES,
        "imus"   : {"file": "imus.npy", "dtype": "float16", "shape": [len(NAMES), T, 9], "fields": fields},
        "pose"   : {"file": "pose.npy", "dtype": "float32", "shape": [len(NAMES), T, 7]},
        "time_ns": {"file": "time_ns.npy", "dtype": "int64", "shape": [T]},
    }))
    return tmp_path


def test_clip_reads_as_cache(clip):
    seq  = Unitree_G1_SuperOdom_IMU_Processed(str(clip))
    data = seq[:]
    imus = torch.from_numpy(np.load(clip / "imus.npy")).float()
    assert seq.kind == "sim" and len(seq) == T
    assert torch.equal(data.acc[0].float(), imus[..., 0:3])
    assert torch.equal(data.gt_vel[0].float(), imus[..., 6:9])


def test_language_reader_loads_clip(clip):
    data = Humanoid_SuperOdom_NPZ_Processed_Language(str(clip))[:16]
    assert data.acc.size(2) == 16
    assert int(data.lang_feat_idx) == -1

