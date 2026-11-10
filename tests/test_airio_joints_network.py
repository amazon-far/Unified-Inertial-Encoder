"""Contracts of the vendored ``JointsAirIONetworkIMUOnly`` used by the AirIOJoints adapter."""
from dataclasses import dataclass

import pypose as pp
import pytest
import torch

from src.model.airio_joints import _load_imunet

KWARGS = dict(
    k_list=[7, 7], s_list=[3, 3], p_list=[3, 3],
    imu_encoder_channels=[32, 64], joint_encoder_channels=[32, 64],
    joint_channels=["position", "velocity"], fusion_dim=64, fusion_hidden=128,
    gru1_hidden=64, gru2_hidden=128, decoder_hidden=128, bidirectional=True,
    num_imus=1, num_joints=33, imu_input="raw",
)


@dataclass
class _Imu:
    acc : torch.Tensor
    gyro: torch.Tensor


@dataclass
class _Joints:
    joint_time_ns: torch.Tensor
    position     : torch.Tensor
    velocity     : torch.Tensor
    effort       : torch.Tensor | None = None


def _input(inetwork, N: int = 256, rot=None, scale: float = 1.):
    B, J = 2, 33
    g = torch.Generator().manual_seed(0)
    return inetwork.Input(
        imu=_Imu(torch.randn(B, N, 1, 3, generator=g), torch.randn(B, N, 1, 3, generator=g)),
        rot=pp.so3(torch.zeros(B, N, 3)) if rot is None else rot,
        joints=_Joints(
            torch.zeros(B, N, 1, dtype=torch.long),
            scale * torch.randn(B, N, J, 1, generator=g),
            scale * torch.randn(B, N, J, 1, generator=g),
        ),
    )


@pytest.fixture(scope="module")
def net():
    net_cls, inetwork = _load_imunet()
    torch.manual_seed(0)
    return net_cls(**KWARGS).eval(), inetwork


@pytest.mark.parametrize("n", [256, 1024])
def test_output_matches_anchor_grid(net, n):
    model, inetwork = net
    with torch.no_grad():
        out = model(_input(inetwork, N=n))
    m = model.anchor_count(n)
    assert out.vel_b.shape == (2, m, 3)
    assert out.vel_cov_b.shape == (2, m, 3, 3)
    assert model.downsample(torch.zeros(2, n, 3)).shape == (2, m, 3)


def test_attitude_is_ignored(net):
    model, inetwork = net
    rot = pp.so3(torch.randn(2, 256, 3))
    with torch.no_grad():
        assert torch.equal(model(_input(inetwork)).vel_b, model(_input(inetwork, rot=rot)).vel_b)


def test_joint_stream_reaches_output(net):
    model, inetwork = net
    with torch.no_grad():
        assert not torch.equal(model(_input(inetwork)).vel_b, model(_input(inetwork, scale=2.)).vel_b)


def test_covariance_is_spd(net):
    model, inetwork = net
    with torch.no_grad():
        assert (torch.linalg.eigvalsh(model(_input(inetwork)).vel_cov_b) > 0).all()
