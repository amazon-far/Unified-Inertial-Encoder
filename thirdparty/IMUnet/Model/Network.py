"""The typed network contract."""

import typing as tp
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import pypose as pp
import jaxtyping as jt
from beartype import beartype


#: Floor on the diagonal of L. softplus underflows in floating point, and a zero diagonal makes
#: the Gaussian NLL return -inf with NaN gradients; its log-det term actively drives cov logits
#: that way. 1e-6 is far above the fp32 cliff and far below any physical velocity uncertainty.
CHOLESKY_DIAG_FLOOR = 1e-6


def cholesky_cov(
    raw_L: jt.Float[torch.Tensor, "*B 6"],
) -> jt.Float[torch.Tensor, "*B 3 3"]:
    """Assemble ``vel_cov = L @ Lᵀ`` from a 6-channel covariance head output.

    Channels are ``[L_00, L_11, L_22, L_10, L_20, L_21]``. The diagonal goes through ``softplus`` plus
    :data:`CHOLESKY_DIAG_FLOOR`, so the product is strictly positive-definite.
    """
    L = raw_L.new_zeros(*raw_L.shape[:-1], 3, 3)
    L[..., 0, 0] = F.softplus(raw_L[..., 0]) + CHOLESKY_DIAG_FLOOR
    L[..., 1, 1] = F.softplus(raw_L[..., 1]) + CHOLESKY_DIAG_FLOOR
    L[..., 2, 2] = F.softplus(raw_L[..., 2]) + CHOLESKY_DIAG_FLOOR
    L[..., 1, 0] = raw_L[..., 3]
    L[..., 2, 0] = raw_L[..., 4]
    L[..., 2, 1] = raw_L[..., 5]
    return L @ L.transpose(-1, -2)


@tp.runtime_checkable
class ImuStack(tp.Protocol):
    """The IMU fields a network reads; K is the multi-IMU axis (K=1 for a single IMU).

    A Protocol checks presence only; ``Input.__post_init__`` checks that they are tensors.
    """

    acc:  jt.Float[torch.Tensor, "B N K 3"]
    gyro: jt.Float[torch.Tensor, "B N K 3"]


@tp.runtime_checkable
class JointStack(tp.Protocol):
    """The joint fields a network reads; J is the joint axis.

    ``effort`` is None on platforms without joint torque (Unitree G1). A network requesting a None
    channel through ``joint_channels`` gets an error, not a zero-fill.
    """

    joint_time_ns: jt.Int64[torch.Tensor, "B N 1"]
    position:      jt.Float[torch.Tensor, "B N J 1"]
    velocity:      jt.Float[torch.Tensor, "B N J 1"]
    effort:        tp.Optional[jt.Float[torch.Tensor, "B N J 1"]]


class INetwork(torch.nn.Module):
    @jt.jaxtyped(typechecker=beartype)
    @dataclass
    class Input:
        """Network input: ``imu`` plus attitude ``rot``.

        Reading ``rot`` is per-network: ``AirIONetwork`` conditions on it, the ``*IMUOnly`` variants ignore it.
        """
        imu: ImuStack
        rot: jt.Float[pp.LieTensor, "B N 3"]    # so3 rotation R_{w <- b}
        # Proprioceptive joints; None for plain IMU batches.
        joints: tp.Optional[JointStack] = None

        def __post_init__(self) -> None:
            # ImuStack is structural, so check that the attributes really are tensors.
            if not (torch.is_tensor(self.imu.acc) and torch.is_tensor(self.imu.gyro)):
                raise TypeError(
                    f"INetwork.Input.imu must carry tensor acc/gyro, got "
                    f"acc={type(self.imu.acc).__name__}, gyro={type(self.imu.gyro).__name__}"
                )
            if self.joints is not None and not (
                torch.is_tensor(self.joints.position) and torch.is_tensor(self.joints.velocity)
            ):
                raise TypeError(
                    f"INetwork.Input.joints must carry tensor position/velocity, got "
                    f"position={type(self.joints.position).__name__}, "
                    f"velocity={type(self.joints.velocity).__name__}"
                )

    @jt.jaxtyped(typechecker=beartype)
    @dataclass
    class Output:
        """Network prediction in the **body frame**; rotate by ``R_{w<-b}`` before comparing with world values."""
        vel_b    : jt.Float[torch.Tensor, "B N 3"]
        vel_cov_b: jt.Float[torch.Tensor, "B N 3 3"]

    def forward(self, x: Input) -> Output: ...

    def anchor_count(self, n_frames: int) -> int:
        """Number of output anchors backed by a real input frame.

        ``forward`` and ``downsample`` both return exactly this many rows.
        """
        ...

    def downsample(
        self,
        x: jt.Float[torch.Tensor, "*B N C"],
        *,
        repeat_tail: bool = False,
    ) -> jt.Float[torch.Tensor, "*B M C"]:
        """Slice ``x`` onto the output anchor grid, returning ``anchor_count(N)`` real rows.

        ``repeat_tail=True`` repeats the final row out to the raw conv output length; those rows are
        fabricated and must not reach a training target or a metric.
        """
        ...

    @property
    def stride(self) -> int:
        """Total temporal stride between consecutive output anchors in input frames."""
        ...

    @property
    def start_index(self) -> int:
        """Input-frame index of output anchor 0 (the end-of-segment + 1 anchor)."""
        ...
