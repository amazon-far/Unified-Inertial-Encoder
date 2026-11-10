"""``FoundationIMU_Ablation_Joint``: FoundationIMU on a pelvis IMU plus joint input, where each
joint enters the jagged pack as a pseudo-device run."""
from __future__ import annotations

import torch
import typing as tp
import pypose as pp
import jaxtyping as jt

from ..interface.human      import CANONICAL_NAMES
from ..interface.model      import ITokenizer, FoundationIMU_Output, FRAME_CLS
from ..interface.data_imu   import IMUData_BaseTrait_Jagged, IMUData_Jagged
from ..interface.data_joint import JointTrait_Jagged

from .model               import FoundationIMU
from .tokenizer           import ContinuousKernel_Conv1DTokenizer, _insert_cls_token
from .template.continuous import ContinuousEncoder
from .template.stride     import Conv1D


#: Site ids of the joint pseudo-devices start right after the canonical IMU sites.
JOINT_SITE_BASE = len(CANONICAL_NAMES)


class JointAware_CK_Tokenizer(ContinuousKernel_Conv1DTokenizer):
    """Continuous-kernel tokenizer that encodes joint pseudo-device runs (site id >= ``JOINT_SITE_BASE``)
    with a dedicated ``joint_encoder`` and joint-site embeddings; decoders are inherited."""

    def __init__(
        self,
        output_dim    : int,
        window_span_ms: float,
        activation    : Conv1D.Support_Act,
        num_taps      : int,
        micro_split   : float = 8.,
        num_trunk_features: int = 1,
        num_joint_sites: int = 11,
    ):
        super().__init__(output_dim, window_span_ms, activation, num_taps, micro_split, num_trunk_features)
        self.num_joint_sites = int(num_joint_sites)
        self.joint_encoder = ContinuousEncoder(
            dim_list   =[6, output_dim // 2, output_dim],
            timespan_ms=[window_span_ms / micro_split, window_span_ms],
            activation =activation,
            transpose  =False,
            num_taps   =num_taps,
        )
        # Canonical IMU sites + J joint sites (ndim=3 keeps it off Muon, like the base).
        self.device_embed = torch.nn.Parameter(torch.zeros((1, JOINT_SITE_BASE + self.num_joint_sites, output_dim)))

    def forward(
        self,
        data: IMUData_BaseTrait_Jagged,
        mask: jt.Bool[torch.Tensor, "pack_BNT"]
    ) -> ITokenizer.Token_Output:
        # ContinuousKernel_Conv1DTokenizer.forward with the differences marked JOINT. Both encoders
        # share one schedule, so token rows line up exactly.
        imu_data = torch.cat([data.acc, data.gyro], dim=-1)

        pack   = mask.shape[0]
        offn   = data.offsets_N
        device = imu_data.device

        time_ns    = data.time_ns[:, 0]
        row        = torch.arange(pack, device=device)
        dev_of_row = torch.searchsorted(offn[1:].contiguous(), row, right=True)
        ref_ns     = time_ns[offn[:-1]][dev_of_row]
        t_ms       = (time_ns - ref_ns).float() * 1e-6

        schedule, r2t = self.imu_encoder.build_schedule(t_ms, offn)
        features, _   = self.imu_encoder(imu_data, t_ms, offn, schedule=schedule, index=r2t)
        # JOINT: encode every run with the joint encoder as well, keep it where the run is a joint.
        feat_joint, _ = self.joint_encoder(imu_data, t_ms, offn, schedule=schedule, index=r2t)

        self._t_ms, self._b_offset = t_ms, offn
        self._schedule, self._r2t  = schedule, r2t
        n_tok = features.shape[0]

        mask_tok           = t_ms.new_zeros(n_tok).index_add_(0, r2t, mask.to(t_ms.dtype)) > 0
        dev_of_tok         = dev_of_row.new_zeros(n_tok); dev_of_tok[r2t] = dev_of_row
        sensor_of_tok      = torch.zeros(n_tok, dtype=torch.long, device=device)
        sensor_of_tok[r2t] = data.sensor_id[:, 0].long()
        cnt_tok            = t_ms.new_zeros(n_tok).index_add_(0, r2t, torch.ones_like(t_ms))
        time_tok           = t_ms.new_zeros(n_tok).index_add_(0, r2t, t_ms) / cnt_tok.clamp_min(1)

        # JOINT: a token is a joint token iff its run's site id lies in the joint range.
        is_joint = sensor_of_tok >= JOINT_SITE_BASE
        features = torch.where(is_joint.unsqueeze(-1), feat_joint, features)

        N             = offn.numel() - 1
        tok_per_dev   = torch.bincount(dev_of_tok, minlength=N)
        tok_offsets_N = torch.cat([offn.new_zeros(1), torch.cumsum(tok_per_dev, 0)])
        dev_at_b      = torch.searchsorted(offn, data.offsets_B)
        tok_offsets_B = tok_offsets_N[dev_at_b]

        features[mask_tok] = self.mask_embed.to(features.dtype).reshape(-1)
        features           = features + self.device_embed[0][sensor_of_tok]

        frame_tok = (time_tok / self.window_span_ms).floor().long()

        features, time_tok, mask_tok, frame_tok, tok_offsets_B, tok_offsets_N, self._base_rows, self._cls_rows = \
            _insert_cls_token(self.cls_embed, features, time_tok, mask_tok, frame_tok, tok_offsets_B, tok_offsets_N)

        return ITokenizer.Token_Output(
            feature =features, time_ms =time_tok, mask=mask_tok,
            offset_b=tok_offsets_B, offset_n=tok_offsets_N, frame=frame_tok,
        )


class FoundationIMU_Ablation_Joint(FoundationIMU):
    """FoundationIMU on a pelvis IMU + joint batch. Sample-space outputs come back on the original
    (pelvis) rows, so losses and evaluators need no changes; joint runs carry no ground truth."""

    def __init__(self, *args, num_joint_sites: int = 11, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_joint_sites = int(num_joint_sites)

    @staticmethod
    def joint_signal(joint_q: torch.Tensor, joint_dq: torch.Tensor, kind: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-row joint channels for the ``acc`` / ``gyro`` slots, ``(P, J, 3)`` each; revolute
        encoders are zero-padded."""
        match kind:
            case "so3":
                pos = pp.SO3(joint_q.float()).Log().tensor()                     # rotation vector (rad)
                vel = joint_dq.float()                                           # child-frame rate (rad/s)
            case "revolute":
                pad = lambda x: torch.cat([x.float(), x.new_zeros(*x.shape[:-1], 2, dtype=torch.float32)], dim=-1)
                pos, vel = pad(joint_q), pad(joint_dq)
            case _:
                raise ValueError(kind)
        return pos, vel

    def augment(self, x: IMUData_BaseTrait_Jagged, mask: torch.Tensor) -> tuple[IMUData_Jagged, torch.Tensor, torch.Tensor]:
        """Interleave ``J`` joint pseudo-device runs after each sample's pelvis run.

        Returns the augmented batch, its mask, and ``pelvis_rows``, the augmented row of every original row.
        """
        assert isinstance(x, JointTrait_Jagged), f"{type(self).__name__} needs a joint-carrying batch, got {type(x).__name__}"
        assert torch.equal(x.offsets_N, x.offsets_B), \
            "FoundationIMU_Ablation_Joint expects exactly one IMU device (the pelvis) per sample"
        J = x.joint_q.shape[1]
        assert J == self.num_joint_sites, f"model built for {self.num_joint_sites} joint sites, batch carries {J}"

        device  = x.acc.device
        P       = x.acc.shape[0]
        B       = x.offsets_B.numel() - 1
        rep     = 1 + J
        T_b     = x.offsets_B.diff()
        samp    = torch.repeat_interleave(torch.arange(B, device=device), T_b)
        local   = torch.arange(P, device=device) - x.offsets_B[:-1][samp]
        base    = rep * x.offsets_B[:-1][samp]                                    # [P] sample start in the augmented pack
        pelvis_rows = base + local
        # joint j of original row r sits at base + (j+1)*T_b + local
        joint_rows = base.unsqueeze(1) + (torch.arange(1, rep, device=device) * T_b[samp].unsqueeze(1)) + local.unsqueeze(1)  # [P, J]

        pos, vel = self.joint_signal(x.joint_q, x.joint_dq, x.joint_kind)
        P_aug = rep * P

        def scatter(pelvis_val: torch.Tensor, joint_val: torch.Tensor | None, fill: float = 0.0) -> torch.Tensor:
            out = pelvis_val.new_full((P_aug, *pelvis_val.shape[1:]), fill)
            out[pelvis_rows] = pelvis_val
            if joint_val is None:
                out[joint_rows.flatten()] = pelvis_val.repeat_interleave(J, dim=0)
            else:
                out[joint_rows.flatten()] = joint_val.reshape(P * J, *joint_val.shape[2:])
            return out

        dt = x.acc.dtype
        acc  = scatter(x.acc,  pos.to(dt))
        gyro = scatter(x.gyro, vel.to(dt))
        sensor_id = scatter(x.sensor_id, (JOINT_SITE_BASE + torch.arange(J, device=device))
                            .to(torch.uint8).view(1, J, 1).expand(P, J, 1))
        offsets_N = torch.cat([x.offsets_B.new_zeros(1), torch.repeat_interleave(T_b, rep).cumsum(0)])
        offsets_B = rep * x.offsets_B

        zeros3 = x.acc.new_zeros(P_aug, 3)
        aug = IMUData_Jagged(
            time_ns      = scatter(x.time_ns, None),
            acc          = acc,
            gyro         = gyro,
            correct_mask = torch.zeros(P_aug, dtype=torch.bool, device=device),
            correct_acc  = acc.clone(),
            correct_gyro = gyro.clone(),
            gravity         = torch.repeat_interleave(x.gravity,         rep, dim=0),
            correct_gravity = torch.repeat_interleave(x.correct_gravity, rep, dim=0),
            acc_bias     = zeros3,
            gyro_bias    = zeros3.clone(),
            offsets_B    = offsets_B,
            offsets_N    = offsets_N,
            sensor_id    = sensor_id,
            model        = x.model,
            kind         = x.kind,
        )
        mask_aug = torch.zeros(P_aug, dtype=torch.bool, device=device)
        mask_aug[pelvis_rows] = mask
        return aug, mask_aug, pelvis_rows

    def forward(self, x: IMUData_BaseTrait_Jagged, mask: jt.Bool[torch.Tensor, "pack_BNT"]) -> FoundationIMU_Output:  # type: ignore[override]
        aug, mask_aug, pelvis_rows = self.augment(x, mask)
        out = super().forward(aug, mask_aug)
        # Sample-space outputs back on the original rows; token-space fields stay on the augmented pack.
        return FoundationIMU_Output(
            trunk_feature    = out.trunk_feature,
            mask_feature     = out.mask_feature,
            offsets_B_feat   = out.offsets_B_feat,
            offsets_N_feat   = out.offsets_N_feat,
            dec_correct_acc  = out.dec_correct_acc[pelvis_rows],
            dec_correct_gyro = out.dec_correct_gyro[pelvis_rows],
            dec_body_vel     = out.dec_body_vel[pelvis_rows],
            dec_lang_feature = out.dec_lang_feature,
            lang_time_ns     = out.lang_time_ns,
            time_ns          = x.time_ns,
            mask_samp        = mask,
            offsets_B_samp   = x.offsets_B,
            offsets_N_samp   = x.offsets_N,
        )
