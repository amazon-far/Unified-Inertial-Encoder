"""Metric interface and row-weighted pooling of per-take statistics."""
import math
import typing as tp
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from rich.table import Table
from rich.console import Console

from .sequence import ISequence
from .data_imu import IMUData_LabelTrait_Jagged
from .model import FoundationIMU_Output



@dataclass(frozen=True)
class MetricPart:
    """One metric's sufficient statistics on a single take, so takes can be pooled.

    ``total`` sums the per-item error over ``count`` items; ``root`` marks RMSE-style metrics,
    which accumulate squares and take the square root once at the end.
    """
    total: float
    count: int
    root : bool

    @property
    def value(self) -> float:
        """The reduced metric; NaN when nothing contributed, so such takes drop out of a pool."""
        if self.count == 0:
            return float("nan")
        mean = self.total / self.count
        return math.sqrt(mean) if self.root else mean


def merge_parts(parts: tp.Iterable[tp.Mapping[str, MetricPart]]) -> dict[str, MetricPart]:
    """Row-weighted union of per-take statistics; a key missing from a take does not contribute."""
    out: dict[str, MetricPart] = {}
    for part in parts:
        for key, item in part.items():
            prev = out.get(key)
            if prev is None:
                out[key] = item
            else:
                assert prev.root == item.root, f"{key}: inconsistent reduction across takes"
                out[key] = MetricPart(prev.total + item.total, prev.count + item.count, item.root)
    return out


def pool(parts: tp.Iterable[tp.Mapping[str, MetricPart]]) -> dict[str, float]:
    """Row-weighted pooled metric values: sum totals and counts, then reduce once."""
    return {key: item.value for key, item in merge_parts(parts).items()}


class IEvaluate_Metric:
    def __init__(self, data: ISequence[IMUData_LabelTrait_Jagged], pred: FoundationIMU_Output):
        """The pred is a FoundationIMU_Output w/ batch=1"""
        ...

    @cached_property
    def parts(self) -> dict[str, MetricPart]:
        """This take's sufficient statistics, keyed like ``metric``; subclasses implement this."""
        ...

    @cached_property
    def metric(self) -> dict:
        """This take's reduced metric values."""
        return {key: item.value for key, item in self.parts.items()}

    @property
    def t0_ns(self) -> int:
        """First timestamp (ns since epoch) of the evaluated window."""
        ...

    def rr_blueprint(self) -> rrb.ContainerLike:
        """This metric's blueprint fragment; the caller composes and sends the blueprint."""
        ...

    def rr_log(self) -> None:
        """Log this metric's entities to the active recording."""
        ...

    @tp.final
    def show(self) -> None:
        """Print the metric table and mirror each entry as a rerun text-log row."""
        table = Table(title=type(self).__name__)
        table.add_column("metric", justify="left")
        table.add_column("value", justify="left")

        rr.set_time("ros_time_ns", timestamp=np.datetime64(self.t0_ns, "ns"))
        for key, value in self.metric.items():
            display = f"{value:.6g}" if isinstance(value, float) else str(value)
            table.add_row(key, display)
            rr.log(f"/metric/{type(self).__name__}", rr.TextLog(f"{key} = {display}", level=rr.TextLogLevel.INFO))

        Console().print(table)
