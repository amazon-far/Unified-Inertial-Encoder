"""Continuous convolutions over irregularly timed samples, and the encoder stack built from them."""
import torch
import typing as tp
import itertools
import torch.nn as nn
import jaxtyping as jt

from .stride import MLP



T = tp.TypeVar("T")
_MISSING = object()
# one entry per level: (window_offset [W+1], window_id [pack], u [pack] designated window-relative time)
ContinuousSchedule = list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


def interleave(*iterables: tp.Iterable[T]) -> tp.Generator[T]:
    for data_pack in itertools.zip_longest(*iterables, fillvalue=_MISSING):
        for data in data_pack:
            if data is not _MISSING: yield tp.cast(T, data)


class ContinuousConv1D(nn.Module):
    """Multi-tap continuous convolution over irregular time, the analog of a strided ``nn.Conv1d``.

    A fixed triangular basis pools each window into ``num_taps`` moments, then a ``Linear`` mixes them
    (KPConv-style blending, not CKConv's implicit-MLP kernel).
    """

    def __init__(self, in_dim: int, out_dim: int, num_taps: int):
        super().__init__()
        assert num_taps >= 2, "triangular basis needs >= 2 taps"
        self.in_dim   = in_dim
        self.num_taps = num_taps
        self.proj     = nn.Linear(in_dim * num_taps, out_dim)
        self.width    = 1.0 / (num_taps - 1)   # tap spacing on u in [0, 1]

    def forward(
        self,
        x          : jt.Float[torch.Tensor, "pack_BNT C_in"],
        u          : jt.Float[torch.Tensor, "pack_BNT"     ],   # designated window-relative time in [0, 1)
        window_id  : jt.Int64[torch.Tensor, "pack_BNT"     ],
        num_windows: int,
    ) -> jt.Float[torch.Tensor, "pack_BNW C_out"]:
        assert x.shape[-1] == self.in_dim, f"conv in_dim {self.in_dim} != input C ({x.shape[-1]})"
        W = num_windows

        # Each sample hits only its two adjacent taps; the ones column accumulates the weight mass.
        ui     = u / self.width                                                 # in [0, m-1]
        k0     = ui.floor().clamp_(0, self.num_taps - 2).long()
        frac   = (ui - k0).unsqueeze(-1).to(x.dtype)
        rows   = window_id * self.num_taps
        aug    = torch.cat([x, x.new_ones(x.shape[0], 1)], dim=-1)
        buf    = x.new_zeros(W * self.num_taps, self.in_dim + 1)
        buf.index_add_(0, rows + k0,     (1.0 - frac) * aug)
        buf.index_add_(0, rows + k0 + 1, frac * aug)
        pooled = buf[:, :self.in_dim] / buf[:, self.in_dim:].clamp_min(1e-6)    # [W*m, in] weighted mean
        return self.proj(pooled.reshape(W, self.num_taps * self.in_dim))


class ContinuousTransposeConv1D(nn.Module):
    """Transpose of :class:`ContinuousConv1D`: ``num_taps`` learned maps ``W_k`` read the coarse token and
    the triangular basis interpolates per sample, ``x_i = (1-f) W_{k0} y + f W_{k0+1} y``."""

    def __init__(self, in_dim: int, out_dim: int, num_taps: int):
        super().__init__()
        assert num_taps >= 2, "triangular basis needs >= 2 taps"
        self.in_dim   = in_dim
        self.num_taps = num_taps
        self.proj     = nn.Linear(in_dim, out_dim * num_taps)   # W_1..W_m stacked
        self.width    = 1.0 / (num_taps - 1)

    def forward(
        self,
        y               : jt.Float[torch.Tensor, "W C_in"],
        u               : jt.Float[torch.Tensor, "pack_BT"],   # designated window-relative time in [0, 1)
        token_of_sample : jt.Int64[torch.Tensor, "pack_BT"],
    ) -> jt.Float[torch.Tensor, "pack_BT C_out"]:
        assert y.shape[-1] == self.in_dim, f"conv in_dim {self.in_dim} != input C ({y.shape[-1]})"
        W    = y.shape[0]
        taps = self.proj(y).reshape(W * self.num_taps, -1)

        ui   = u / self.width                                   # in [0, m-1]
        k0   = ui.floor().clamp_(0, self.num_taps - 2).long()
        frac = (ui - k0).unsqueeze(-1).to(taps.dtype)
        rows = token_of_sample * self.num_taps
        return (1.0 - frac) * taps[rows + k0] + frac * taps[rows + k0 + 1]


class ContinuousEncoder(nn.Module):
    def __init__(
        self,
        dim_list      : list[int],          # channel flow C_0 -> C_1 -> ... -> C_n
        timespan_ms   : list[float],        # window span pooled/upsampled at each conv (len == len(dim_list)-1)
        activation    : MLP.Support_Act,
        transpose     : bool,               # pool (ContinuousConv1D) or upsample (ContinuousTransposeConv1D)?
        num_taps      : int = 8,            # taps per conv
    ):
        """Continuous analog of ``Conv1D``. Pass ``dim_list``/``timespan_ms`` in flow order (in -> out);
        a transpose stack rebuilds the same schedule, so it inverts its pooling twin."""
        super().__init__()
        assert len(timespan_ms) == len(dim_list) - 1
        self.transpose   = transpose
        # Schedule is built fine -> coarse; a transpose stack is handed the mirror order, so flip it back.
        self.timespan_ms = timespan_ms[::-1] if transpose else timespan_ms

        conv_cls = ContinuousTransposeConv1D if transpose else ContinuousConv1D
        convs    = [conv_cls(dim_list[i], dim_list[i + 1], num_taps) for i in range(len(dim_list) - 1)]
        acts     = [MLP.act_fn(activation) for _ in range(len(convs) - 1)]
        self.module = nn.ModuleList(interleave(convs, acts))

    def forward(
        self,
        x: jt.Float[torch.Tensor, "pack_BT C"],
        t_ms: jt.Float[torch.Tensor, "pack_BT"],
        b_offset: jt.Int64[torch.Tensor, "B+1"],
        schedule: ContinuousSchedule | None = None,
        index: jt.Int64[torch.Tensor, "pack_BT"] | None = None,
    ) -> tuple[jt.Float[torch.Tensor, "pack_BX C_out"], jt.Int64[torch.Tensor, "pack_BT"]]:
        """``t_ms``/``b_offset`` describe the raw sample grid. Also returns the raw-sample -> token row map."""
        if schedule is None:
            schedule, index = self.build_schedule(t_ms, b_offset)
        else:
            assert len(schedule) == len(self.timespan_ms), \
                f"schedule has {len(schedule)} levels, encoder expects {len(self.timespan_ms)}"
            if index is None:
                index = self._compose_raw_to_token_index(t_ms.shape[0], schedule, t_ms.device)

        # Pooling walks the fine->coarse schedule low->high, a transpose stack high->low.
        levels     = range(len(self.timespan_ms))
        level_iter = iter(reversed(levels) if self.transpose else levels)
        for layer in self.module:
            if isinstance(layer, ContinuousConv1D):
                window_offset, window_id, u = schedule[next(level_iter)]
                x = layer(x, u, window_id, window_offset.numel() - 1)
            elif isinstance(layer, ContinuousTransposeConv1D):
                _, window_id, u = schedule[next(level_iter)]
                x = layer(x, u, window_id)
            else:
                x = layer(x)
        return x, index

    def build_schedule(
        self,
        t_ms: jt.Float[torch.Tensor, "pack_BT"],
        b_offset: jt.Int64[torch.Tensor, "B+1"],
    ) -> tuple[ContinuousSchedule, jt.Int64[torch.Tensor, "pack_BT"]]:
        schedule: ContinuousSchedule = []
        index = torch.arange(t_ms.shape[0], device=t_ms.device)         # raw row -> token row
        for timespan in self.timespan_ms:
            window_offset, b_offset, t_window, window_id, u = self._segment_windows(t_ms, b_offset, timespan)
            schedule.append((window_offset, window_id, u))
            index = window_id[index]
            t_ms  = t_window
        return schedule, index

    @staticmethod
    def _compose_raw_to_token_index(
        num_raw : int,
        schedule: ContinuousSchedule,
        device  : torch.device,
    ) -> jt.Int64[torch.Tensor, "pack_BT"]:
        index = torch.arange(num_raw, device=device)
        for _, window_id, _ in schedule:
            index = window_id[index]
        return index

    @staticmethod
    def _segment_windows(
        t_ms       : jt.Float[torch.Tensor, "pack_BT"],
        b_offset   : jt.Int64[torch.Tensor, "B+1"],
        timespan_ms: float,
    ) -> tuple[jt.Int64[torch.Tensor, "W+1"], jt.Int64[torch.Tensor, "B+1"], jt.Float[torch.Tensor, "W"], jt.Int64[torch.Tensor, "pack_BT"], jt.Float[torch.Tensor, "pack_BT"]]:
        """Bin each B-sample's samples into fixed ``timespan_ms`` windows, dropping empty bins.

        Returns the CSR window offsets, rebuilt B offsets, each window's mean sample time, the per-sample
        window id, and ``u in [0, 1)``: the fraction within the FIXED bin (not the first/last-sample
        span), so the kernel is stationary.
        """
        P      = t_ms.shape[0]
        device = t_ms.device
        B      = b_offset.numel() - 1

        samp_per_b = b_offset[1:] - b_offset[:-1]
        b_id       = torch.repeat_interleave(torch.arange(B, device=device), samp_per_b)
        b_start    = t_ms[b_offset[:-1]]                                                   # [B] window-grid origin
        ratio      = (t_ms - b_start[b_id]) / timespan_ms
        bin_local  = ratio.floor()
        u          = (ratio - bin_local).to(t_ms.dtype)
        bin_local  = bin_local.to(torch.int64)

        # A change in (b_id, bin_local) marks a new window, so windows never cross a B boundary.
        boundary      = torch.ones(P, dtype=torch.bool, device=device)
        boundary[1:]  = (b_id[1:] != b_id[:-1]) | (bin_local[1:] != bin_local[:-1])
        window_id     = torch.cumsum(boundary.to(torch.int64), 0) - 1
        counts        = torch.bincount(window_id)
        window_offset = torch.cat([b_offset.new_zeros(1), torch.cumsum(counts, 0)])

        b_of_window  = b_id[window_offset[:-1]]
        win_per_b    = torch.bincount(b_of_window, minlength=B)
        b_offset_new = torch.cat([b_offset.new_zeros(1), torch.cumsum(win_per_b, 0)])

        # segment_reduce (not scatter), relative to window start: this feeds the next level's binning, so
        # non-deterministic atomics could drift independently rebuilt schedules to different token counts.
        t0       = t_ms[window_offset[:-1]]
        rel_t    = t_ms - t0[window_id]
        t_window = t0 + torch.segment_reduce(rel_t, "mean", lengths=counts, axis=0)
        return window_offset, b_offset_new, t_window, window_id, u
