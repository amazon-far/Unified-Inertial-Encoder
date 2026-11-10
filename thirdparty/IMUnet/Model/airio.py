"""AirIO velocity networks: with attitude, IMU-only, and IMU plus joints."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pypose as pp

import jaxtyping as jt
from typing import cast, Sequence

from .Encoder.CNN  import MotionEncoder
from .Network      import INetwork, cholesky_cov


#: Which IMU stream a network reads: ``acc``/``gyro`` as measured, or
#: ``corrected_acc``/``corrected_gyro``.
IMU_STREAMS = ("raw", "corrected")


def select_imu(imu, stream: str):
    """Return the (acc, gyro) pair named by ``stream``."""
    if stream == "corrected":
        return (
            cast(torch.Tensor, imu.corrected_acc),
            cast(torch.Tensor, imu.corrected_gyro),
        )
    assert stream == "raw"
    return imu.acc, imu.gyro


class AirIONetwork(INetwork):
    """Strided CNN + GRU body-frame velocity network with a Cholesky covariance head.

    Inputs: body-frame acc/gyro (gravity not removed) and rot = GT R_{w<-b}.Log(). Outputs are
    body-frame at each strided anchor. Multi-IMU (K>1) is fused in the conv stem's channel dim.
    """

    def __init__(
        self,
        k_list:               Sequence[int],
        s_list:               Sequence[int],
        p_list:               Sequence[int],
        imu_encoder_channels: Sequence[int],
        ori_encoder_channels: Sequence[int],
        fusion_dim:           int,
        gru1_hidden:          int,
        gru2_hidden:          int,
        decoder_hidden:       int,
        bidirectional:        bool,
        num_imus:             int,
        imu_input:            str,
    ) -> None:
        super().__init__()
        if imu_input not in IMU_STREAMS:
            raise ValueError(f"imu_input must be one of {IMU_STREAMS}, got {imu_input!r}")
        self.imu_input = str(imu_input)

        # The input channel is computed (6 * num_imus), so encoder lists carry only output widths.
        if not (
            len(imu_encoder_channels)
            == len(ori_encoder_channels)
            == len(k_list)
            == len(s_list)
            == len(p_list)
        ):
            raise ValueError(
                "expected equal encoder-layout lengths, got "
                f"imu_encoder_channels={list(imu_encoder_channels)}, "
                f"ori_encoder_channels={list(ori_encoder_channels)}, "
                f"k_list={list(k_list)}, s_list={list(s_list)}, p_list={list(p_list)}"
            )
        if len(k_list) != 2:
            raise ValueError(
                f"AirIONetwork currently assumes a 2-layer CNN encoder; got k_list={list(k_list)}. "
                "The end-of-segment anchor formula in `downsample` is hand-derived for L=2."
            )
        if num_imus < 1:
            raise ValueError(f"num_imus must be >= 1, got {num_imus}")
        self.num_imus = num_imus

        imu_c_list = [6 * num_imus, *imu_encoder_channels] # acc + gyro
        ori_c_list = [3, *ori_encoder_channels] # so(3) tangent of GT R_{w<-b} (axis-angle)

        self.k_list = tuple(k_list)
        self.s_list = tuple(s_list)
        self.p_list = tuple(p_list)

        self.feature_encoder = MotionEncoder(
            c_list=imu_c_list, k_list=k_list, s_list=s_list, p_list=p_list,
        )
        self.ori_encoder = MotionEncoder(
            c_list=ori_c_list, k_list=k_list, s_list=s_list, p_list=p_list,
        )

        concat_dim = imu_encoder_channels[-1] + ori_encoder_channels[-1]
        self.fcn2       = nn.Linear(concat_dim, fusion_dim)
        self.batchnorm2 = nn.BatchNorm1d(fusion_dim)
        self.gelu       = nn.GELU()

        factor    = 2 if bidirectional else 1
        self.gru1 = nn.GRU(input_size=fusion_dim          , hidden_size=gru1_hidden, num_layers=1, batch_first=True, bidirectional=bidirectional)
        self.gru2 = nn.GRU(input_size=gru1_hidden * factor, hidden_size=gru2_hidden, num_layers=1, batch_first=True, bidirectional=bidirectional)

        self.veldecoder = nn.Sequential(
            nn.Linear(gru2_hidden * factor, decoder_hidden),
            nn.GELU(),
            nn.Linear(decoder_hidden, 3),
        )

        # Cholesky-factor head; see ``cholesky_cov`` for the channel layout.
        self.velcov_decoder = nn.Sequential(
            nn.Linear(gru2_hidden * factor, decoder_hidden),
            nn.GELU(),
            nn.Linear(decoder_hidden, 6),
        )

    @property
    def stride(self) -> int:
        """Total temporal stride between consecutive output anchors (= prod(s_list))."""
        return self.s_list[0] * self.s_list[1]

    @property
    def start_index(self) -> int:
        """Input-frame index of output anchor 0 (matches the slice in ``downsample``)."""
        return (self.k_list[0] - self.p_list[0]) + self.s_list[0] * (self.k_list[1] - 1 - self.p_list[1])

    def conv_output_length(self, n_frames: int) -> int:
        """Raw number of values the conv stack emits for ``n_frames`` inputs; may exceed ``anchor_count``."""
        length = n_frames
        for k, s, p in zip(self.k_list, self.s_list, self.p_list):
            length = (length + 2 * p - k) // s + 1
        return length

    def anchor_count(self, n_frames: int) -> int:
        """Number of output anchors backed by a real input frame.

        Anchor ``i`` aligns to input frame ``start_index + i*stride``. Conv outputs past the end of the
        input have no ground truth and are not anchors; their number is not always 1.
        """
        s_idx = self.start_index
        if n_frames <= s_idx:
            raise ValueError(
                f"{type(self).__name__}: need more than start_index={s_idx} frames "
                f"to form an anchor; got n_frames={n_frames}."
            )
        return (n_frames - 1 - s_idx) // self.stride + 1

    def downsample(
        self,
        x: jt.Float[torch.Tensor, "*B N C"],
        *,
        repeat_tail: bool = False,
    ) -> jt.Float[torch.Tensor, "*B M C"]:
        """Slice (*B, N, C) onto the output anchors: x[..., s_idx::stride, :].

            s_idx  = (k_0 - p_0) + s_0 * (k_1 - 1 - p_1)     # end-of-segment + 1
            stride = s_0 * s_1

        ``repeat_tail=True`` repeats the final row out to ``conv_output_length(N)``; those rows are
        fabricated and must not reach a training target or a metric. Supports Tensor and LieTensor.
        """
        selected = x[..., self.start_index::self.stride, :]

        # Tripwire: a kernel/stride change must not silently shift every target by a frame.
        expected = self.anchor_count(x.shape[-2])
        assert selected.shape[-2] == expected
        if not repeat_tail:
            return selected

        n_phantom = self.conv_output_length(x.shape[-2]) - selected.shape[-2]
        if n_phantom <= 0:
            return selected

        repeats = (1,) * (x.ndim - 2) + (n_phantom, 1)
        tail = x[..., -1:, :].repeat(*repeats)
        if isinstance(x, pp.LieTensor):
            # torch.cat may drop the LieTensor subclass.
            raw = torch.cat(
                [selected.as_subclass(torch.Tensor), tail.as_subclass(torch.Tensor)],
                dim=-2,
            )
            return pp.LieTensor(raw, ltype=x.ltype)
        return torch.cat([selected, tail], dim=-2)

    def _encode_imu(self, x: "INetwork.Input") -> torch.Tensor:
        """Encode K IMUs, flattened into the conv channel dim, to ``(B, T, imu_encoder_channels[-1])``."""
        acc, gyro = select_imu(x.imu, self.imu_input)
        imu_feat = torch.cat([acc, gyro], dim=-1)                # (B, N, K, 6)
        B, N, K, _ = imu_feat.shape
        if K != self.num_imus:
            raise ValueError(
                f"{type(self).__name__}: the conv stem was built for num_imus="
                f"{self.num_imus} (in_channels={6 * self.num_imus}), but the batch "
                f"carries K={K} IMUs. The K axis is folded into the channel dim, so "
                "the two must match exactly."
            )
        imu_feat = imu_feat.reshape(B, N, K * 6)                 # (B, N, 6K)
        # Conv1d wants (B, C, N); transpose back to (B, T, C) after.
        return self.feature_encoder(imu_feat.transpose(-1, -2)).transpose(-1, -2)

    def _encode_rot(self, x: "INetwork.Input") -> torch.Tensor:
        """Encode the so(3) attitude-tangent stream to ``(B, T, ori_encoder_channels[-1])``."""
        rot_feat = x.rot.tensor()                                # (B, N, 3)
        return self.ori_encoder(rot_feat.transpose(-1, -2)).transpose(-1, -2)

    def encode(self, x: "INetwork.Input") -> torch.Tensor:
        """Encode and fuse all input streams to ``(B, T, fusion_dim)``; subclasses override this."""

        f = torch.cat([self._encode_imu(x), self._encode_rot(x)], dim=-1)           # (B, T, concat_dim)
        f = self.fcn2(f)                                                            # (B, T, fusion_dim)
        f = self.batchnorm2(f.transpose(-1, -2)).transpose(-1, -2)
        return self.gelu(f)

    def forward(self, x: INetwork.Input) -> INetwork.Output:
        N = x.imu.acc.shape[1]
        return self.decode_and_trim(self.encode(x), N)

    def decode_and_trim(
        self,
        f: jt.Float[torch.Tensor, "B T C"],
        N: int,
    ) -> INetwork.Output:
        """GRU trunk, vel/cov decode and anchor trim.

        ``f`` is ``(B, T, fusion_dim)``; ``N`` is the number of input frames, needed for the trim.
        """
        self.gru1.flatten_parameters()
        self.gru2.flatten_parameters()
        f, _ = self.gru1(f)                                                        # (B, T, gru1_hidden)
        f, _ = self.gru2(f)                                                        # (B, T, gru2_hidden)

        vel     = self.veldecoder(f)                                               # (B, T, 3)
        vel_cov = cholesky_cov(self.velcov_decoder(f.detach()))                    # (B, T, 3, 3) — stop-grad shared features

        # Drop conv outputs past the end of the input: they come mostly from zero-padding and have
        # no ground truth. This keeps forward and downsample on one grid.
        M = self.anchor_count(N)
        return self.Output(vel_b=vel[:, :M], vel_cov_b=vel_cov[:, :M])


# The per-joint measurements a loader may carry. A network's `joint_channels` is an ordered
# subset; the order sets the conv-stem layout, so checkpoints depend on it.
JOINT_CHANNELS: tuple[str, ...] = ("position", "velocity", "effort")


def _encode_joint_stack(
    encoder: MotionEncoder,
    x: "INetwork.Input",
    owner: str,
    channels: tuple[str, ...],
) -> torch.Tensor:
    """Encode the joint stack to ``(B, T, joint_encoder_channels[-1])``.

    Flattens ``(B, N, J, C) -> (B, N, C*J)`` with joint j at channels C*j..C*j+C-1; this order is the
    checkpoint layout. A channel carried as None is an error, never a zero-fill.
    """
    assert x.joints is not None, f"{owner} requires INetwork.Input.joints; got None"
    parts = []
    for name in channels:
        # Unknown names and unmeasured channels raise the same error.
        field = getattr(x.joints, name, None)
        if field is None:
            raise ValueError(
                f"{owner}: joint_channels requests {name!r}, which this batch does not "
                f"provide (unknown field, or carried as None). Valid channels are "
                f"{list(JOINT_CHANNELS)}; drop it from joint_channels, or use a dataset "
                f"that measures it."
            )
        parts.append(field)
    joint_feat = torch.cat(parts, dim=-1)                                # (B, N, J, C)
    B, N, J, C = joint_feat.shape
    joint_feat = joint_feat.reshape(B, N, J * C)                         # (B, N, C*J)
    # Conv1d wants (B, C, N); transpose back to (B, T, C) after.
    return encoder(joint_feat.transpose(-1, -2)).transpose(-1, -2)


class AirIONetworkIMUOnly(AirIONetwork):
    """Body-frame AirIO network without orientation input; ``Input.rot`` is ignored.

    Checkpoints are not interchangeable with ``AirIONetwork``: no ``ori_encoder``, different ``fcn2`` width.
    """

    def __init__(
        self,
        k_list:               Sequence[int],
        s_list:               Sequence[int],
        p_list:               Sequence[int],
        imu_encoder_channels: Sequence[int],
        fusion_dim:           int,
        gru1_hidden:          int,
        gru2_hidden:          int,
        decoder_hidden:       int,
        bidirectional:        bool,
        num_imus:             int,
        imu_input:            str,
    ) -> None:
        # Skip AirIONetwork.__init__, which builds the ori branch.
        INetwork.__init__(self)
        if imu_input not in IMU_STREAMS:
            raise ValueError(f"imu_input must be one of {IMU_STREAMS}, got {imu_input!r}")
        self.imu_input = str(imu_input)

        if not (len(imu_encoder_channels) == len(k_list) == len(s_list) == len(p_list)):
            raise ValueError(
                f"expected len(imu_encoder_channels) == len(k_list) == len(s_list) == len(p_list), got "
                f"imu_encoder_channels={list(imu_encoder_channels)}, k_list={list(k_list)}, "
                f"s_list={list(s_list)}, p_list={list(p_list)}"
            )
        if len(k_list) != 2:
            raise ValueError(
                f"AirIONetworkIMUOnly assumes a 2-layer CNN encoder; got k_list={list(k_list)}. "
                "The end-of-segment anchor formula in `downsample` is hand-derived for L=2."
            )
        if num_imus < 1:
            raise ValueError(f"num_imus must be >= 1, got {num_imus}")
        self.num_imus = num_imus

        self.k_list = tuple(k_list)
        self.s_list = tuple(s_list)
        self.p_list = tuple(p_list)

        self.feature_encoder = MotionEncoder(
            c_list=[6 * num_imus, *imu_encoder_channels],  # acc + gyro per IMU
            k_list=k_list, s_list=s_list, p_list=p_list,
        )

        self.fcn2       = nn.Linear(imu_encoder_channels[-1], fusion_dim)
        self.batchnorm2 = nn.BatchNorm1d(fusion_dim)
        self.gelu       = nn.GELU()

        factor    = 2 if bidirectional else 1
        self.gru1 = nn.GRU(input_size=fusion_dim          , hidden_size=gru1_hidden, num_layers=1, batch_first=True, bidirectional=bidirectional)
        self.gru2 = nn.GRU(input_size=gru1_hidden * factor, hidden_size=gru2_hidden, num_layers=1, batch_first=True, bidirectional=bidirectional)

        self.veldecoder = nn.Sequential(
            nn.Linear(gru2_hidden * factor, decoder_hidden),
            nn.GELU(),
            nn.Linear(decoder_hidden, 3),
        )
        self.velcov_decoder = nn.Sequential(
            nn.Linear(gru2_hidden * factor, decoder_hidden),
            nn.GELU(),
            nn.Linear(decoder_hidden, 6),
        )

    def encode(self, x: "INetwork.Input") -> torch.Tensor:
        """Encode the IMU stream only; ``x.rot`` is ignored."""
        f = self._encode_imu(x)                                                     # (B, T, imu_encoder_channels[-1])
        f = self.fcn2(f)                                                            # (B, T, fusion_dim)
        f = self.batchnorm2(f.transpose(-1, -2)).transpose(-1, -2)
        return self.gelu(f)


class JointsAirIONetworkIMUOnly(AirIONetworkIMUOnly):
    """Fuse body-frame IMU and joint streams without orientation input; ``Input.rot`` is ignored."""

    def __init__(
        self,
        k_list:                 Sequence[int],
        s_list:                 Sequence[int],
        p_list:                 Sequence[int],
        imu_encoder_channels:   Sequence[int],
        joint_encoder_channels: Sequence[int],
        joint_channels:         Sequence[str],
        fusion_dim:             int,
        fusion_hidden:          int,
        gru1_hidden:            int,
        gru2_hidden:            int,
        decoder_hidden:         int,
        bidirectional:          bool,
        num_imus:               int,
        num_joints:             int,
        imu_input:              str,
    ) -> None:
        super().__init__(
            k_list=k_list, s_list=s_list, p_list=p_list,
            imu_encoder_channels=imu_encoder_channels, fusion_dim=fusion_dim,
            gru1_hidden=gru1_hidden, gru2_hidden=gru2_hidden,
            decoder_hidden=decoder_hidden, bidirectional=bidirectional,
            num_imus=num_imus, imu_input=imu_input,
        )

        if num_joints < 1:
            raise ValueError(f"num_joints must be >= 1, got {num_joints}")
        self.num_joints = num_joints
        self.joint_channels = tuple(joint_channels)

        if len(joint_encoder_channels) != len(k_list):
            raise ValueError(
                f"expected len(joint_encoder_channels) == len(k_list), got "
                f"joint_encoder_channels={list(joint_encoder_channels)}, "
                f"k_list={list(k_list)}"
            )
        # Same k/s/p as the IMU encoder, so both streams land on one time grid.
        self.joint_encoder = MotionEncoder(
            c_list=[len(self.joint_channels) * num_joints, *joint_encoder_channels],
            k_list=k_list, s_list=s_list, p_list=p_list,
        )

        # 3-layer MLP, not a single Linear: joint<->IMU coupling is more nonlinear than IMU<->attitude.
        concat_dim = imu_encoder_channels[-1] + joint_encoder_channels[-1]
        self.fusion_mlp = nn.Sequential(
            nn.Linear(concat_dim,    fusion_hidden),
            nn.GELU(),
            nn.Linear(fusion_hidden, fusion_hidden),
            nn.GELU(),
            nn.Linear(fusion_hidden, fusion_dim),
        )

    def encode(self, x: "INetwork.Input") -> torch.Tensor:
        """Fuse IMU and joints to (B, T, fusion_dim); ``x.rot`` is ignored."""
        f = torch.cat(
            [
                self._encode_imu(x),
                _encode_joint_stack(
                    self.joint_encoder, x, type(self).__name__, self.joint_channels,
                ),
            ],
            dim=-1,
        )                                                                           # (B, T, concat_dim)
        f = self.fusion_mlp(f)                                                      # (B, T, fusion_dim)
        f = self.batchnorm2(f.transpose(-1, -2)).transpose(-1, -2)
        return self.gelu(f)
