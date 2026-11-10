"""Sliding and random windowing of IMU sequences, in sample or millisecond units."""

import torch
import random
import typing as tp

from ...interface.sequence import ISequence
from ...interface.data_imu import IMUData_BaseTrait, IMUData_BaseTrait_Jagged, TokenUnit


def _sorted_timestamps(source: ISequence) -> torch.Tensor:
    """Return ``source.timestamps``, asserting order: ``searchsorted`` is silently wrong otherwise."""
    time_ns = source.timestamps
    assert bool((time_ns.diff() >= 0).all()), \
        f"{type(source).__name__}.timestamps must be non-decreasing for ms windowing"
    return time_ns


class SlidingWindow_IMU(ISequence[IMUData_BaseTrait_Jagged]):
    def __init__(self, window_size: int, window_unit: TokenUnit, step_size: int, source: ISequence[IMUData_BaseTrait]) -> None:
        super().__init__()
        self.window_size = window_size
        self.window_unit: TokenUnit = window_unit
        self.step_size   = step_size
        self.source      = source
        
        self.ranges = self._generate_start_points()

    def _generate_start_points(self) -> list[tuple[int, int]]:
        match self.window_unit:
            case "ms":
                # Fixed time span, variable sample count: window k starts at t0 + k*step.
                time_ns   = _sorted_timestamps(self.source)
                window_ns = int(self.window_size * 1_000_000)
                step_ns   = int(self.step_size   * 1_000_000)
                if time_ns.numel() == 0:
                    return []
                if int(time_ns[-1] - time_ns[0]) < window_ns:
                    # Shorter than one window: emit it whole. Partial tails of longer sources drop.
                    return [(0, time_ns.numel())]
                starts_ns = torch.arange(int(time_ns[0]), int(time_ns[-1]) - window_ns + 1, step_ns)
                start_idx = torch.searchsorted(time_ns, starts_ns)
                end_idx   = torch.searchsorted(time_ns, starts_ns + window_ns)
                return [(s, e - s) for s, e in zip(start_idx.tolist(), end_idx.tolist()) if e > s]
            case "sample":
                # int(): YAML sizes may be floats, which range() rejects.
                # `+ 1`: start = len - window is a valid full window.
                window, step = int(self.window_size), int(self.step_size)
                if 0 < len(self.source) < window:
                    # Shorter than one window: emit it whole.
                    return [(0, len(self.source))]
                return [(_, window) for _ in range(0, len(self.source) - window + 1, step)]
            case _:
                raise ValueError(f"Unexpected window_unit of {self.window_unit}")

    def __len__(self) -> int:
        return len(self.ranges)

    def __getitem__(self, index: slice | int) -> IMUData_BaseTrait_Jagged:
        if isinstance(index, int):
            index = slice(index, index + 1)

        ranges: list[tuple[int, int]] = self.ranges[index]
        
        samples = [self.source[start : start + size].as_jagged() for start, size in ranges]
        return type(samples[0]).collate(samples)


class RandomWindow_IMU(ISequence[IMUData_BaseTrait_Jagged]):
    def __init__(self, window_size: int, window_unit: TokenUnit, source: ISequence[IMUData_BaseTrait]):
        super().__init__()
        self.window_size = window_size
        self.window_unit: TokenUnit = window_unit
        self.source      = source

        # `max_start` / `num_window` live on the unit's own axis: ns offset from t0 for "ms",
        # sample index for "sample".
        match self.window_unit:
            case "ms":
                self.window_ns = int(window_size * 1_000_000)
                self.time_ns   = _sorted_timestamps(source)
                self._n_source = self.time_ns.numel()
                extent, window = (int(self.time_ns[-1] - self.time_ns[0]) if self._n_source > 0 else -1), self.window_ns
            case "sample":
                self._n_source = len(source)
                extent, window = self._n_source, int(window_size)
            case _:
                raise ValueError(f"Unexpected window_unit of {self.window_unit}")

        # A non-empty source shorter than one window still yields one whole-source window.
        self.max_start  = extent - window
        self.num_window = self.max_start // window + 1 if self.max_start >= 0 else int(self._n_source > 0)

    def __len__(self) -> int:
        return self.num_window

    def _draw_range(self) -> tuple[int, int]:
        if self.max_start < 0:
            return 0, self._n_source
        match self.window_unit:
            case "ms":
                # Redraw windows landing in a recording gap; start == t0 always yields a sample,
                # so this terminates.
                while True:
                    start_ns = int(self.time_ns[0]) + random.randint(0, self.max_start)
                    s = int(torch.searchsorted(self.time_ns, start_ns))
                    e = int(torch.searchsorted(self.time_ns, start_ns + self.window_ns))
                    if e > s:
                        return s, e - s
            case "sample":
                return random.randint(0, self.max_start), int(self.window_size)

    def __getitem__(self, index: slice | int) -> IMUData_BaseTrait_Jagged:
        if isinstance(index, int):
            index = slice(index, index + 1)

        ranges  = [self._draw_range() for _ in range(*index.indices(len(self)))]
        samples = [self.source[start : start + size].as_jagged() for start, size in ranges]
        return type(samples[0]).collate(samples)
