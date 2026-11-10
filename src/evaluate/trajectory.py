"""Trajectory metric (ATE/RTE/ROE) from predicted body velocity and integrated rotation."""
import typing as tp

import torch
import pypose as pp
import rerun as rr
import rerun.blueprint as rrb
from functools import cached_property

from ..interface.evaluate import IEvaluate_Metric, MetricPart
from ..interface.human import CANONICAL_NAMES
from ..interface.data_imu import IMUData_BaseTrait_Jagged, IMUData_LabelTrait_Jagged, IMUData_Jagged
from ..interface.model import FoundationIMU_Output
from ..model.integrator import rk4_preintegrate, apply_initial_state, integrate_trajectory



class Trajectory_Metric(IEvaluate_Metric):
    def __init__(
        self,
        data: IMUData_LabelTrait_Jagged,
        pred: FoundationIMU_Output,
        gyro_source: tp.Literal["measured", "denoised", "clean"] = "measured",
    ):
        # `data` must be the same collated batch `inference` ran over, so packed rows align 1:1
        # with pred's sample pack. Everything runs on pred's device.
        device = pred.dec_body_vel.device
        batch  = data
        # Trait check: lang-carrying batches are siblings of IMUData_w_GT_Jagged, not subclasses.
        assert isinstance(batch, IMUData_BaseTrait_Jagged) and isinstance(batch, IMUData_LabelTrait_Jagged), \
            f"Trajectory_Metric needs gt_pose/gt_vel alongside the base fields, got {type(batch).__name__}"
        batch = batch.to(device, None)   # move only; keep dtypes
        assert isinstance(batch, IMUData_LabelTrait_Jagged)   # re-narrow for the checker

        time_ns  = batch.time_ns[:, 0]
        pred_vel = pred.dec_body_vel.detach().to(torch.float64)

        # The eval pack is window-major, but integrators treat each offsets_N run as one continuous
        # stream, so reorder to device-major, time-sorted runs.
        sensor_id  = batch.sensor_id[:, 0]
        perm_parts = []
        boundaries = [0]
        names      = []
        for sid in sensor_id.unique().tolist():
            rows = (sensor_id == sid).nonzero().flatten()
            perm_parts.append(rows[time_ns[rows].argsort()])
            boundaries.append(boundaries[-1] + rows.numel())
            names.append(CANONICAL_NAMES[sid] if sid < len(CANONICAL_NAMES) else f"id{sid}")
        perm      = torch.cat(perm_parts)
        offsets_N = torch.tensor(boundaries, dtype=torch.int64, device=device)
        N         = offsets_N.numel() - 1
        run       = torch.repeat_interleave(torch.arange(N, device=device), offsets_N.diff())

        gt_pose = pp.SE3(batch.gt_pose.tensor()[perm].to(torch.float64))

        # Preintegrate `gyro_source`, then compose each run onto its initial GT rotation.
        # "measured": raw gyro; "denoised": model denoise head; "clean": oracle upper bound.
        # Only rotations are consumed, so the acc slot is inert.
        rot_gyro = {
            "measured": lambda: batch.gyro,
            "denoised": lambda: pred.dec_correct_gyro.detach().to(batch.gyro.dtype),
            "clean"   : lambda: batch.correct_gyro,
        }[gyro_source]()
        source = IMUData_Jagged(
            time_ns      = batch.time_ns[perm],
            acc          = batch.acc[perm],
            gyro         = batch.gyro[perm],
            correct_mask = batch.correct_mask[perm],
            correct_acc  = batch.correct_acc[perm],
            correct_gyro = rot_gyro[perm],
            gravity         = torch.zeros(N, 3, device=device),
            correct_gravity = torch.zeros(N, 3, device=device),
            acc_bias     = batch.acc_bias[perm],
            gyro_bias    = batch.gyro_bias[perm],
            offsets_B    = offsets_N[[0, -1]],
            offsets_N    = offsets_N,
            sensor_id    = batch.sensor_id[perm],
            model        = batch.model,
            kind         = batch.kind,
        )
        R_wb_0  = pp.SO3(gt_pose.rotation().tensor()[offsets_N[:-1]])
        # Only the rotations are used, so v_b_0 and gravity_w are zero.
        _, R_wb = apply_initial_state(
            rk4_preintegrate(source),
            R_wb_0    = R_wb_0,
            v_b_0     = torch.zeros(N, 3, device=device),
            gravity_w = torch.zeros(N, 3, device=device),
        )

        # Predicted body velocity rotated by GT rotation (isolates velocity error) vs integrated
        # rotation (full dead-reckoning). integrate_trajectory starts each run at the world origin,
        # so the GT translation is rebased to match.
        self.gt_rot              = gt_pose.rotation()
        self.pred_rot            = R_wb
        self.pred_xyz_w_gt_rot   = integrate_trajectory(pred_vel[perm], self.gt_rot, batch.time_ns[perm], offsets_N)
        self.pred_xyz_w_inte_rot = integrate_trajectory(pred_vel[perm], self.pred_rot, batch.time_ns[perm], offsets_N)
        self.gt_xyz              = gt_pose.translation() - gt_pose.translation()[offsets_N[:-1][run]]

        self.time_ns   = time_ns[perm].cpu()
        self.offsets_N = offsets_N.cpu()
        self._device_names = names   # run order (offsets_N order)

    @property
    def t0_ns(self) -> int:
        return int(self.time_ns.min())

    @cached_property
    def parts(self) -> dict[str, MetricPart]:
        # GPU-parallel equivalents of evo's APE/RPE: the translation norm of
        # E_i = (Q_i^-1 Q_j)^-1 (P_i^-1 P_j) reduces to the difference of body-frame steps.
        # No trajectory alignment: runs share the GT initial state, so aligning would hide drift.
        # ROE applies only to the integrated rotation; the other trajectory carries GT rotations.
        device = self.gt_xyz.device
        off    = self.offsets_N.to(device)
        run    = torch.repeat_interleave(torch.arange(off.numel() - 1, device=device), off.diff())
        t      = self.time_ns.to(device)

        # ROE pairs: consecutive same-run frames (1-frame).
        f0 = torch.arange(self.gt_xyz.shape[0] - 1, device=device)[run[:-1] == run[1:]]
        f1 = f0 + 1

        # RTE pairs: consecutive anchors on a 1 s grid per run, dropping pairs a gap stretched by >10%.
        s0, s1 = [], []
        for d in range(off.numel() - 1):
            s, e   = int(off[d]), int(off[d + 1])
            anchor = torch.searchsorted(t[s:e], torch.arange(int(t[s]), int(t[e - 1]) + 1, 1_000_000_000, device=device)) + s
            s0.append(anchor[:-1])
            s1.append(anchor[1:])
        s0, s1 = torch.cat(s0), torch.cat(s1)
        keep   = ((t[s1] - t[s0]).double() * 1e-9 - 1.0).abs() < 0.1
        s0, s1 = s0[keep], s1[keep]

        def step_b(xyz: torch.Tensor, rot: pp.LieTensor, i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
            # Per-pair step in the frame of pose i == translation part of P_i^-1 P_j.
            return pp.SO3(rot.tensor()[i]).Inv() @ (xyz[j] - xyz[i])

        rmse    = lambda e: MetricPart(float(e.square().sum()), e.numel(), root=True)
        ref_b   = step_b(self.gt_xyz, self.gt_rot, s0, s1)
        rel_ref = pp.SO3(self.gt_rot.tensor()[f0]).Inv()   @ pp.SO3(self.gt_rot.tensor()[f1])
        rel_est = pp.SO3(self.pred_rot.tensor()[f0]).Inv() @ pp.SO3(self.pred_rot.tensor()[f1])

        return {
            "ate/pred_w_gt_rot"     : rmse((self.pred_xyz_w_gt_rot   - self.gt_xyz).norm(dim=-1)),
            "ate/pred_w_inte_rot"   : rmse((self.pred_xyz_w_inte_rot - self.gt_xyz).norm(dim=-1)),
            "rte_1s/pred_w_gt_rot"  : rmse((step_b(self.pred_xyz_w_gt_rot,   self.gt_rot,   s0, s1) - ref_b).norm(dim=-1)),
            "rte_1s/pred_w_inte_rot": rmse((step_b(self.pred_xyz_w_inte_rot, self.pred_rot, s0, s1) - ref_b).norm(dim=-1)),
            "roe_1f/pred_w_inte_rot": rmse((rel_ref.Inv() @ rel_est).Log().norm(dim=-1).rad2deg()),
        }

    def rr_blueprint(self) -> rrb.ContainerLike:
        # Plane views are pinhole projections of the same 3D entities.
        white = [255, 255, 255]
        def plane_view(name: str, plane: str):
            return rrb.Spatial2DView(
                name       = plane.upper(),
                origin     = f"/trajectory/{name}/proj/{plane}",
                contents   = f"/trajectory/{name}/3d/**",
                background = white,
            )
        return rrb.Tabs(
            *[
                rrb.Grid(
                    rrb.Spatial3DView(name="3D", origin=f"/trajectory/{name}/3d", background=white),
                    plane_view(name, "xy"),
                    plane_view(name, "yz"),
                    plane_view(name, "xz"),
                    grid_columns=2,
                    name=name,
                )
                for name in self._device_names
            ],
            name="Trajectory",
        )

    def rr_log(self) -> None:
        colors = {
            "gt"             : [160, 160, 160],
            "pred_w_gt_rot"  : [ 28, 225,  64],
            "pred_w_inte_rot": [255,  36,  36],
        }
        trajectories = {
            "gt"             : self.gt_xyz.cpu(),
            "pred_w_gt_rot"  : self.pred_xyz_w_gt_rot.cpu(),
            "pred_w_inte_rot": self.pred_xyz_w_inte_rot.cpu(),
        }
        rr.log("/trajectory", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

        # Camera axes as world-frame columns [right, down, forward] (pinhole RDF convention).
        CAM_AXES = {
            "xy": [[1, 0,  0], [0, -1, 0], [0,  0, -1]],   # top-down  : +x right, +y up
            "yz": [[0, 0, -1], [1,  0, 0], [0, -1,  0]],   # from +x   : +y right, +z up
            "xz": [[1, 0,  0], [0,  0, 1], [0, -1,  0]],   # from -y   : +x right, +z up
        }
        RES = 1600

        for d, name in enumerate(self._device_names):
            s, e    = int(self.offsets_N[d]), int(self.offsets_N[d + 1])
            time    = self.time_ns[s:e].numpy().astype("int64").view("datetime64[ns]")
            indexes = [rr.TimeColumn("ros_time_ns", timestamp=time)]

            # A far-away, long-focal pinhole per plane approximates an orthographic projection.
            xyz_all = torch.cat([xyz[s:e] for xyz in trajectories.values()])
            center  = (xyz_all.amax(0) + xyz_all.amin(0)) / 2
            radius  = float((xyz_all.amax(0) - xyz_all.amin(0)).norm() / 2) + 1e-3
            dist    = 20.0 * radius
            for plane, axes in CAM_AXES.items():
                forward = torch.tensor(axes, dtype=torch.float64)[:, 2]
                cam     = f"/trajectory/{name}/proj/{plane}"
                rr.log(cam, rr.Transform3D(translation=(center - dist * forward).numpy(), mat3x3=axes), static=True)
                rr.log(cam, rr.Pinhole(resolution=[RES, RES], focal_length=0.95 * (RES / 2) * dist / radius), static=True)

            for key, xyz in trajectories.items():
                path = f"/trajectory/{name}/3d/{key}"
                # Static full path; a time-indexed marker follows the shared timeline.
                rr.log(path, rr.LineStrips3D([xyz[s:e].numpy()], colors=[colors[key]], radii=-1.0), static=True)
                rr.log(f"{path}/now", rr.Points3D.from_fields(colors=[colors[key]], radii=-3.0), static=True)
                rr.send_columns(f"{path}/now", indexes=indexes, columns=rr.Points3D.columns(positions=xyz[s:e].numpy()))
