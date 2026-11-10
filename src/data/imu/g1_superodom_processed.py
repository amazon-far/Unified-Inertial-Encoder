"""Reader for real-G1 format-v3 clip caches, including the published clips."""

import typing as tp

from ...interface.human     import T_CANONICAL_NAMES
from ...interface.data_imu  import Loss_Flags

from .nymeria_processed      import Nymeria_XSens_Processed_IMU
from ..humanoid.g1_superodom import G1_WholeBody_Link_Mapping


class Unitree_G1_SuperOdom_IMU_Processed(Nymeria_XSens_Processed_IMU):
    """Reads a real-G1 format-v3 clip cache, defaulting to the ``G1_WholeBody_Link_Mapping`` sites.

    The clips at https://huggingface.co/datasets/Open-IO/humanoid-g1-odometry are this same cache,
    already windowed and mount-corrected; they name the packed array ``imus`` and omit ``kind``.
    """
    
    # Ground truth is lower quality, so prefer label-free losses.
    APPLICABLE_LOSS = Loss_Flags.JEPA \
                    | Loss_Flags.Vel_SSIO

    def __init__(self, path: str, target_sensors: tp.Sequence[T_CANONICAL_NAMES] | None = None):
        if target_sensors is None:
            target_sensors = tuple(G1_WholeBody_Link_Mapping.keys())
        super().__init__(path, target_sensors)

    def _read_spec(self) -> dict[str, tp.Any]:
        spec = super()._read_spec()
        if "measure" not in spec:
            spec["measure"] = spec["imus"]
        spec.setdefault("kind", "sim")
        return spec
