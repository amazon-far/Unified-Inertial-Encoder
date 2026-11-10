"""Per-device body-frame velocity metric with rerun logging."""
import torch
import rerun as rr
import rerun.blueprint as rrb
from functools import cached_property

from ..interface.evaluate import IEvaluate_Metric, MetricPart
from ..interface.human import CANONICAL_NAMES
from ..interface.data_imu import IMUData_BaseTrait_Jagged, IMUData_LabelTrait_Jagged
from ..interface.model import FoundationIMU_Output



class Velocity_Metric(IEvaluate_Metric):
    def __init__(self, batch: IMUData_LabelTrait_Jagged, pred: FoundationIMU_Output):
        # `batch` must be the same collated batch `inference` ran over, so packed rows align 1:1
        # with pred's sample pack. gt_vel and dec_body_vel are both body-frame.
        # Trait check: lang-carrying batches are siblings of IMUData_w_GT_Jagged, not subclasses.
        assert isinstance(batch, IMUData_BaseTrait_Jagged) and isinstance(batch, IMUData_LabelTrait_Jagged), \
            f"Velocity_Metric needs gt_vel alongside the base fields, got {type(batch).__name__}"

        self.gt_vel    = batch.gt_vel.float()
        self.pred_vel  = pred.dec_body_vel.detach().float().cpu()
        self.time_ns   = batch.time_ns[:, 0]
        self.sensor_id = batch.sensor_id[:, 0]

    @cached_property
    def parts(self) -> dict[str, MetricPart]:
        if self.gt_vel.numel() == 0: return {}

        def entries(err: torch.Tensor) -> dict[str, MetricPart]:
            # Rotation-invariant: the error norm is the same in body and world frames.
            sq, l2 = err.pow(2).sum(dim=-1), err.norm(dim=-1)
            n      = err.shape[0]
            return {
                "vel_mse" : MetricPart(float(sq.sum()), n, root=False),  # mean squared Euclidean error, (m/s)^2
                "vel_rmse": MetricPart(float(sq.sum()), n, root=True),
                "vel_l1"  : MetricPart(float(l2.sum()), n, root=False),  # mean Euclidean error magnitude, m/s
            }

        err    = self.gt_vel - self.pred_vel
        result = entries(err)
        for sid, name in self._device_names.items():
            result |= {f"{name}/{key}": part for key, part in entries(err[self.sensor_id == sid]).items()}
        return result

    @property
    def t0_ns(self) -> int:
        return int(self.time_ns.min())

    @cached_property
    def _device_names(self) -> dict[int, str]:
        return {
            sid: (CANONICAL_NAMES[sid] if sid < len(CANONICAL_NAMES) else f"id{sid}")
            for sid in self.sensor_id.unique().tolist()
        }

    def rr_blueprint(self) -> rrb.ContainerLike:
        return rrb.Tabs(
            *[
                rrb.Vertical(
                    rrb.TimeSeriesView(name="Velocity", origin=f"/velocity/{name}/vel"),
                    rrb.TimeSeriesView(name="Error"   , origin=f"/velocity/{name}/vel_error"),
                    row_shares=[2, 1],
                    name=name,
                )
                for name in self._device_names.values()
            ],
            name="Velocity",
        )

    def rr_log(self) -> None:
        gt_colors   = [[105,   0,  0], [ 0,  95,  0], [ 0,   0, 115]]
        pred_colors = [[255,  36, 36], [28, 225, 64], [64, 128, 255]]

        def series_lines(names: list[str], colors: list[list[int]], widths: list[float]):
            return rr.SeriesLines(
                names=names, colors=colors, widths=widths,
                aggregation_policy=rr.components.AggregationPolicy.Off,
            )

        for sid, name in self._device_names.items():
            rows     = self.sensor_id == sid
            order    = self.time_ns[rows].argsort()
            time     = self.time_ns[rows][order].numpy().astype("int64").view("datetime64[ns]")
            gt, pred = self.gt_vel[rows][order], self.pred_vel[rows][order]
            indexes  = [rr.TimeColumn("ros_time_ns", timestamp=time)]

            vel_path = f"/velocity/{name}/vel"
            rr.log(
                vel_path,
                series_lines(
                    names =[f"gt_{a}" for a in "xyz"] + [f"pred_{a}" for a in "xyz"],
                    colors=gt_colors + pred_colors,
                    widths=[1.0] * 3 + [1.6] * 3,
                ),
                static=True,
            )
            rr.send_columns(
                vel_path, indexes=indexes,
                columns=rr.Scalars.columns(scalars=torch.cat([gt, pred], dim=-1).numpy()),
            )

            err_path = f"/velocity/{name}/vel_error"
            rr.log(
                err_path,
                series_lines(names=[f"err_{a}" for a in "xyz"], colors=pred_colors, widths=[1.2] * 3),
                static=True,
            )
            rr.send_columns(
                err_path, indexes=indexes,
                columns=rr.Scalars.columns(scalars=(pred - gt).numpy()),
            )
