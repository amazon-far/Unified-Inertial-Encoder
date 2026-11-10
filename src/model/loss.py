"""Training losses: denoising, velocity, language and integrated-trajectory terms, and their
gradient-balanced composition."""
import math
import torch
import pypose as pp
import jaxtyping as jt
import torch.nn.functional as F

from contextlib import nullcontext
from typing import Any, NotRequired, Sequence, TypedDict, Literal

from .model import FoundationIMU
from .integrator import Preint_Jagged, euler_preintegrate, rk4_preintegrate, integrate_trajectory
from ..interface.data_imu import (
    IMUData_Jagged, IMUData_w_GT_Jagged, IMUData_w_GT_Lang_Jagged, Loss_Flags, TokenUnit,
)
from ..interface.model import ILossFunc, FoundationIMU_Output
from ..utility.jagged import quaternion_multiply


class LossCompose_AutoBalance(ILossFunc):
    class FuncSpec(TypedDict):
        func : ILossFunc
        grad : bool          # whether this loss enters the gradient/total (False -> metric-only)
        name : str           # unique id
        # 'auto' (default): gradient-derived factor normalized to the anchor; a number: fixed multiplier.
        scale: NotRequired[float | Literal["auto"]]

    def __init__(
        self,
        funcs: Sequence[FuncSpec],
        anchor: str,
        min_lambda: float,
        max_lambda: float,
        ema_decay : float
    ):
        super().__init__()
        assert anchor in {f['name'] for f in funcs}, f"{anchor=} not found in funcs!"

        self.funcs  = torch.nn.ModuleDict({func["name"] : func["func"] for func in funcs})
        self.active = {func["name"] : func["grad"] for func in funcs}
        self.scale  = {func["name"] : func.get("scale", "auto") for func in funcs}
        self.anchor = anchor
        self.range  = (min_lambda, max_lambda)
        self.ema    = ema_decay

        for name, scale in self.scale.items():
            assert scale == "auto" or isinstance(scale, (int, float)), \
                f"scale for {name} must be 'auto' or a number, got {scale!r}"

        self.factor: dict[str, float] | None = None
    
    def normalization_factor(self, model: FoundationIMU, loss: dict[str, torch.Tensor]):
        # Only 'auto' losses are probed, plus the anchor as the norm reference.
        grads = {
            name : torch.autograd.grad(
                l, model.grad_probe.weight,
                retain_graph=True, create_graph=False, allow_unused=True,
            )[0]
            for name, l in loss.items()
            if self.active[name] and (self.scale[name] == "auto" or name == self.anchor)
        }
        # A gated-off loss has a None or exactly-zero probe grad. Mark it NaN, not the 1e-6 floor that would
        # blow the factor to the clamp ceiling, so it is neither folded into the EMA nor logged. An absent
        # anchor makes every factor NaN.
        raw_grad_norm_dict = {
            name : (float("nan") if g is None or not bool(g.any()) else g.detach().norm(p=2).item())
            for name, g in grads.items()
        }
        grad_factor = {
            name : (float("nan") if (g is None or grads[self.anchor] is None)
                    else raw_grad_norm_dict[self.anchor] / (raw_grad_norm_dict[name] + 1e-6))
            for name, g in grads.items()
        }
        # Keep the prior value for a loss absent this step, so NaN never poisons its EMA.
        prev = self.factor or {}
        self.factor = dict(prev)
        for name, gf in grad_factor.items():
            if math.isnan(gf): continue
            self.factor[name] = (
                (self.ema * prev[name]) + ((1 - self.ema) * gf)
                if (name in prev) else gf
            )
        self.factor[self.anchor] = 1.0
        self.factor = {
            name : max(self.range[0], min(value, self.range[1]))
            for name, value in self.factor.items()
        }
        
        # For logging only; forward() reads the NaN-free self.factor.
        step_factor = {
            name : (float("nan") if math.isnan(gf) else self.factor[name])
            for name, gf in grad_factor.items()
        }
        return raw_grad_norm_dict, step_factor
    
    def get_extra_state(self) -> Any:
        return {"factor": self.factor}

    def set_extra_state(self, state: Any) -> None:
        self.factor = state["factor"]
    
    def forward(
        self,
        model    : FoundationIMU,
        data     : IMUData_w_GT_Jagged,
        pred     : FoundationIMU_Output
    ) -> tuple[torch.Tensor, dict[str, float]]:
        loss: torch.Tensor | None = None
        loss_dict, metric = dict(), dict()

        for name, func in self.funcs.items():
            i_loss, i_metric   = func(model, data, pred)
            metric            |= i_metric
            loss_dict[name]    = i_loss

        any_auto = any(self.active[name] and self.scale[name] == "auto" for name in self.funcs)
        if torch.is_grad_enabled() and any_auto:
            raw_norm, step_factor = self.normalization_factor(model, loss_dict)
            assert self.factor is not None

            metric |= {
                f"grad_norm/{name}": value for name, value in raw_norm.items()
                if not math.isnan(value)
            } | {
                f"s_factor/{name}" : value for name, value in step_factor.items()
                if not math.isnan(value)
            }

        for name, func in self.funcs.items():
            if not self.active[name]: continue

            if (scale := self.scale[name]) == "auto":
                factor = 1.0 if self.factor is None else self.factor.get(name, 1.0)
            else:
                factor = float(scale)
            scaled_loss = factor * loss_dict[name]
            loss        = loss + scaled_loss if (loss is not None) else scaled_loss

        assert loss is not None
        
        metric["loss"] = loss.item()
        return loss, metric


class Denoise_Loss(ILossFunc):
    
    LABEL = Loss_Flags.Denoise_L2
    
    def __init__(self, masked_weight: float | None, unmasked_weight: float | None):
        """set weight to 'None' to disable that specific path of compute."""
        super().__init__()
        self.masked_weight   = masked_weight
        self.unmasked_weight = unmasked_weight

    def forward(
        self,
        model    : FoundationIMU,
        data     : IMUData_w_GT_Jagged,
        pred     : FoundationIMU_Output
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Skip when no device opted in. The zero must stay graph-connected to this loss's head: a detached
        # zero leaves the head without gradients on this rank only, which deadlocks DDP's bucket allreduce
        # unless find_unused_parameters is on.
        applies = (data.labels.to(torch.int64) & int(self.LABEL)) != 0
        if not bool(applies.any()):
            return (pred.dec_correct_acc.float().sum() + pred.dec_correct_gyro.float().sum()) * 0.0, {}
        gate = torch.repeat_interleave(applies, data.offsets_N.diff())

        acc_err = (data.correct_acc  - pred.dec_correct_acc).norm(dim=-1)
        gyr_err = (data.correct_gyro - pred.dec_correct_gyro).norm(dim=-1)

        keep = data.correct_mask & gate
        acc_loss_mask = acc_err[pred.mask_samp  & keep].mean()
        gyr_loss_mask = gyr_err[pred.mask_samp  & keep].mean()
        acc_loss_copy = acc_err[~pred.mask_samp & keep].mean()
        gyr_loss_copy = gyr_err[~pred.mask_samp & keep].mean()

        match self.unmasked_weight, self.masked_weight:
            case None, None: raise ValueError("Must enable at least one compute path")
            case None, mw   :
                loss = mw * acc_loss_mask.nan_to_num() + mw * gyr_loss_mask.nan_to_num()
            case cw   , None:
                loss = cw * acc_loss_copy.nan_to_num() + cw * gyr_loss_copy.nan_to_num()
            case cw  , mw  :
                loss = (cw * acc_loss_copy.nan_to_num() + cw * gyr_loss_copy.nan_to_num()) \
                     + (mw * acc_loss_mask.nan_to_num() + mw * gyr_loss_mask.nan_to_num())

        metric   = {"loss" : loss.item()} \
            | ({} if torch.isnan(acc_loss_mask) else {"recon/acc_loss"  : acc_loss_mask.item()}) \
            | ({} if torch.isnan(gyr_loss_mask) else {"recon/gyro_loss" : gyr_loss_mask.item()}) \
            | ({} if torch.isnan(acc_loss_copy) else {"copy/acc_loss"   : acc_loss_copy.item()}) \
            | ({} if torch.isnan(gyr_loss_copy) else {"copy/gyro_loss"  : gyr_loss_copy.item()})

        return loss, metric


class Velocity_Loss(ILossFunc):
    
    LABEL = Loss_Flags.Vel_L2
    
    def __init__(self, masked_weight: float | None, unmasked_weight: float | None, huber_delta: float | None = None):
        """set weight to 'None' to disable that specific path of compute.

        `huber_delta` (m/s) bounds the per-sample penalty (Huber on the error magnitude); `None` keeps
        the plain L1 error magnitude.
        """
        super().__init__()
        self.masked_weight   = masked_weight
        self.unmasked_weight = unmasked_weight
        self.huber_delta     = huber_delta

    def forward(
        self,
        model    : FoundationIMU,
        data     : IMUData_w_GT_Jagged,
        pred     : FoundationIMU_Output
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # gt_vel and dec_body_vel are both body-frame, so they are directly comparable.

        # Graph-connected zero: see Denoise_Loss.
        applies = (data.labels.to(torch.int64) & int(self.LABEL)) != 0
        if not bool(applies.any()):
            return pred.dec_body_vel.float().sum() * 0.0, {}
        gate = torch.repeat_interleave(applies, data.offsets_N.diff())

        mask_samp =  pred.mask_samp[gate]
        copy_samp = ~pred.mask_samp[gate]

        vel_l1 = (data.gt_vel[gate] - pred.dec_body_vel[gate]).norm(dim=-1)

        # Huber drives the gradient; the metrics below still report the plain L1 error.
        vel_grad = vel_l1 if self.huber_delta is None else \
            F.huber_loss(vel_l1, torch.zeros_like(vel_l1), delta=self.huber_delta, reduction="none")

        grad_mask = vel_grad[mask_samp].mean()
        grad_copy = vel_grad[copy_samp].mean()

        match self.unmasked_weight, self.masked_weight:
            case None, None: raise ValueError("Must enable at least one compute path")
            case None, mw   :
                loss = mw * grad_mask.nan_to_num()
            case cw   , None:
                loss = cw * grad_copy.nan_to_num()
            case cw  , mw  :
                loss = cw * grad_copy.nan_to_num() + mw * grad_mask.nan_to_num()

        recon_vel_loss = vel_l1[mask_samp].mean()
        copy_vel_loss  = vel_l1[copy_samp].mean()
        metric  = {"loss" : loss.item()} \
                | ({} if torch.isnan(recon_vel_loss) else {"recon/vel_loss" : recon_vel_loss.item()}) \
                | ({} if torch.isnan(copy_vel_loss)  else {"copy/vel_loss"  : copy_vel_loss.item()})

        return loss, metric


class Lang_CLIP_Loss(ILossFunc):
    """Cosine-distance regression of the per-sample [cls] token onto its CLIP language feature.

    Each sample's target is row ``lang_feat_idx[b]`` of ``lang_feature`` (``-1``: no covering annotation),
    supervised iff that row is valid and the sample opted in via the ``Lang_CLIP`` label bit. Regression,
    not contrastive: the targets are a small redundant vocabulary, so in-batch negatives would be false.
    """

    LABEL = Loss_Flags.Lang_CLIP

    def forward(
        self,
        model    : FoundationIMU,
        data     : IMUData_w_GT_Jagged,
        pred     : FoundationIMU_Output
    ) -> tuple[torch.Tensor, dict[str, float]]:
        assert isinstance(data, IMUData_w_GT_Lang_Jagged), \
            f"Lang_CLIP_Loss requires an IMUData_w_GT_Lang_Jagged batch, got {type(data).__name__}"

        lang_pred = pred.dec_lang_feature[:, 0]                                # [B, C_lang]; T_lang is 1 during training
        device    = lang_pred.device

        # The Lang_CLIP bit is uniform across a sample's devices, so read it off the first device.
        first_dev = torch.searchsorted(data.offsets_N, data.offsets_B[:-1])
        opted_in  = (data.labels[first_dev].to(torch.int64) & int(self.LABEL)) != 0
        has_feat  = data.lang_feat_idx >= 0
        valid     = opted_in & has_feat
        if not bool(valid.any()):
            # Graph-connected zero: see Denoise_Loss.
            return lang_pred.float().sum() * 0.0, {}

        target = data.lang_feature[data.lang_feat_idx[valid]].to(device)
        p      = F.normalize(lang_pred[valid].float(), p=2.0, dim=-1, eps=1e-6)
        t      = F.normalize(target.float(),           p=2.0, dim=-1, eps=1e-6)
        cos    = (p * t).sum(dim=-1)
        loss   = (1.0 - cos).mean()

        # Collapse watch; population std so a single-valid-window batch reports 0, not NaN.
        pred_std = lang_pred[valid].float().std(dim=0, unbiased=False).mean()

        metric = {
            "loss"           : loss.item(),
            "lang/cos"       : cos.mean().item(),
            "lang/n_valid"   : float(int(valid.sum())),
            "lang/pred_std"  : pred_std.item(),
        }
        return loss, metric


# --- Shared jagged-pack machinery for trajectory supervision ---

def _restrict_to_applies(
    data   : IMUData_w_GT_Jagged,
    applies: jt.Bool[torch.Tensor, "n_dev"],
) -> tuple[
    jt.Int64[torch.Tensor, "pack_sub"],
    jt.Int64[torch.Tensor, "n_sub"],
    jt.Int64[torch.Tensor, "n_sub+1"],
]:
    """Restrict the pack to the opted-in device runs: returns ``(sub2full, sub_len, sub_offn)``."""
    lengths  = data.offsets_N.diff()
    keep_row = torch.repeat_interleave(applies, lengths)
    sub2full = keep_row.nonzero(as_tuple=True)[0]
    sub_len  = lengths[applies]
    sub_offn = torch.cat([sub_len.new_zeros(1), sub_len.cumsum(0)])
    return sub2full, sub_len, sub_offn


def _channel_pack(
    data        : IMUData_w_GT_Jagged,
    applies     : jt.Bool [torch.Tensor, "n_dev"],
    sub2full    : jt.Int64[torch.Tensor, "pack_sub"],
    sub_offn    : jt.Int64[torch.Tensor, "n_sub+1"],
    correct_acc : jt.Float[torch.Tensor, "pack_sub 3"],
    correct_gyro: jt.Float[torch.Tensor, "pack_sub 3"],
) -> IMUData_Jagged:
    """The opted-in device runs as a standalone pack for the preintegrator, with the given tensors in
    the corrected-channel slots. It reads only those, ``time_ns`` and the offsets, so the remaining
    fields are placeholders and ``offsets_B`` degenerates to one pseudo-sample.
    """
    return IMUData_Jagged(
        time_ns      = data.time_ns[sub2full],
        acc          = correct_acc,
        gyro         = correct_gyro,
        correct_mask = data.correct_mask[sub2full],
        correct_acc  = correct_acc,
        correct_gyro = correct_gyro,
        gravity         = data.gravity[applies],
        correct_gravity = data.correct_gravity[applies],
        acc_bias     = data.acc_bias [sub2full],
        gyro_bias    = data.gyro_bias[sub2full],
        offsets_B    = torch.stack([sub_offn[0], sub_offn[-1]]),
        offsets_N    = sub_offn,
        sensor_id    = data.sensor_id[sub2full],
        model        = data.model,
        kind         = data.kind,
    )


def _aligned_prefixes(
    preint  : Preint_Jagged,
    sub_offn: jt.Int64[torch.Tensor, "n_sub+1"],
) -> tuple[
    jt.Float[torch.Tensor, "pack_sub 4"],
    jt.Float[torch.Tensor, "pack_sub 3"],
    jt.Float[torch.Tensor, "pack_sub 3"],
    jt.Float[torch.Tensor, "pack_sub 1"],
]:
    """Re-insert each run's omitted initial row so prefix row ``t`` aligns with sample row
    ``t`` of the sub pack: ``(ΔR_0t, Δv_0t, Δp_0t, Δt_0t)`` with identity/zero at ``t = 0``.
    """
    n_sub   = sub_offn.numel() - 1
    pack    = int(sub_offn[-1])
    device  = preint.all_Dv.device
    dtype   = preint.all_Dv.dtype

    run      = torch.repeat_interleave(torch.arange(n_sub, device=device), sub_offn.diff())
    row      = torch.arange(pack, device=device)
    endpoint = row[row != sub_offn[:-1][run]]

    Dr = pp.identity_SO3(pack, device=device, dtype=dtype).tensor()
    Dr = Dr.index_copy(0, endpoint, preint.all_Dr.tensor())
    Dv = torch.zeros(pack, 3, device=device, dtype=dtype).index_copy(0, endpoint, preint.all_Dv)
    Dp = torch.zeros(pack, 3, device=device, dtype=dtype).index_copy(0, endpoint, preint.all_Dp)
    Dt = torch.zeros(pack, 1, device=device, dtype=dtype).index_copy(0, endpoint, preint.all_Dt)
    return Dr, Dv, Dp, Dt


def _grid_windows(
    data    : IMUData_w_GT_Jagged,
    sub2full: jt.Int64[torch.Tensor, "pack_sub"],
    sub_offn: jt.Int64[torch.Tensor, "n_sub+1"],
    horizon : float,
    stride  : float,
    unit    : TokenUnit,
) -> tuple[
    jt.Int64[torch.Tensor, "W"],
    jt.Int64[torch.Tensor, "W"],
    jt.Int64[torch.Tensor, "W"],
]:
    """Fixed-grid supervision windows per device run of the sub pack.

    Returns ``(i, j, dev_of_win)`` in sub-pack row / sub-run coordinates: starts every ``stride``,
    endpoints ``horizon`` later. "ms" walks a per-run time grid (rate-invariant, tolerant of recording
    gaps); "sample" strides raw rows.
    """
    device  = sub2full.device
    n_sub   = sub_offn.numel() - 1
    sub_len = sub_offn.diff()

    def expand(n_win: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # (dev_of_win, within-run window ordinal) for per-run window counts `n_win`.
        base       = torch.cat([n_win.new_zeros(1), n_win.cumsum(0)[:-1]])
        dev_of_win = torch.repeat_interleave(torch.arange(n_sub, device=device), n_win)
        ordinal    = torch.arange(int(n_win.sum()), device=device) - base[dev_of_win]
        return dev_of_win, ordinal

    match unit:
        case "sample":
            W, S  = int(horizon), int(stride)
            n_win = ((sub_len - 1 - W).div(S, rounding_mode="floor") + 1).clamp_min(0)
            dev_of_win, ordinal = expand(n_win)
            i = sub_offn[:-1][dev_of_win] + ordinal * S
            return i, i + W, dev_of_win

        case "ms":
            h_ns, s_ns = round(horizon * 1e6), round(stride * 1e6)
            t     = data.time_ns[sub2full, 0]
            run   = torch.repeat_interleave(torch.arange(n_sub, device=device), sub_len)
            t_rel = t - t[sub_offn[:-1]][run]
            span  = t_rel[sub_offn[1:] - 1]
            n_win = ((span - h_ns).div(s_ns, rounding_mode="floor") + 1).clamp_min(0)
            dev_of_win, ordinal = expand(n_win)

            # Per-run searchsorted through one sorted key: the per-run offset exceeds any within-run time.
            offset = int(span.max()) + h_ns + 1
            key    = t_rel + run * offset
            start  = ordinal * s_ns + dev_of_win * offset
            i = torch.searchsorted(key, start)
            j = torch.searchsorted(key, start + h_ns)
            ok = j > i                                                 # a gap can swallow the whole window
            return i[ok], j[ok], dev_of_win[ok]

        case _:
            raise ValueError(f"Unexpected window unit {unit}")


class _Trajectory_LossBase(ILossFunc):
    """Shared machinery for the integrated-trajectory losses.

    A subclass integrates one trajectory per opted-in device run, seeded from the GT initial state.
    Optimized: fixed-horizon relative increments as drift rates (``/Δt``), RTE and (subclass permitting)
    ROE, since a whole-window absolute error is dominated by the earliest samples. Metric-only:
    whole-window ATE / AOE and per-horizon RTE / ROE RMSEs, matching ``src/evaluate/trajectory.py``.

    Metrics go under ``{METRIC_PREFIX}/``, which must be unique per concrete loss.
    """

    LABEL         = Loss_Flags.EMPTY
    METRIC_PREFIX = ""

    def __init__(
        self,
        horizons   : Sequence[float],
        strides    : Sequence[float] | None,
        unit       : TokenUnit,
        integrator : Literal["rk4", "euler"],
        precision  : Literal["float32", "float64"],
        huber_delta: float | None,
    ):
        super().__init__()
        stride_list = [float(s) for s in strides] if strides is not None else [float(h) for h in horizons]
        assert len(horizons) > 0 and len(stride_list) == len(horizons), \
            f"horizons/strides length mismatch: {list(horizons)} vs {stride_list}"
        assert all(h > 0 for h in horizons) and all(s > 0 for s in stride_list)
        self.horizons        = [float(h) for h in horizons]
        self.strides         = stride_list
        self.unit: TokenUnit = unit
        self.preintegrate    = {"rk4": rk4_preintegrate, "euler": euler_preintegrate}[integrator]
        self.dtype           = {"float32": torch.float32, "float64": torch.float64}[precision]
        self.huber_delta     = huber_delta

    def _zero(self, pred: FoundationIMU_Output) -> torch.Tensor:
        """Graph-connected zero touching exactly the heads the real path trains (see Denoise_Loss)."""
        raise NotImplementedError

    def _integrate(
        self,
        data    : IMUData_w_GT_Jagged,
        pred    : FoundationIMU_Output,
        applies : jt.Bool [torch.Tensor, "n_dev"],
        sub2full: jt.Int64[torch.Tensor, "pack_sub"],
        sub_offn: jt.Int64[torch.Tensor, "n_sub+1"],
        run     : jt.Int64[torch.Tensor, "pack_sub"],
        gt_rot  : jt.Float[pp.LieTensor, "pack_sub 4"],
    ) -> tuple[jt.Float[torch.Tensor, "pack_sub 3"], jt.Float[pp.LieTensor, "pack_sub 4"]]:
        """Per sub-pack row: world position (origin at each run's first row) and attitude ``R̂_wb``."""
        raise NotImplementedError

    def _horizon_term(self, rte_rate: torch.Tensor, roe_rate: torch.Tensor) -> torch.Tensor:
        """One horizon's optimized term from drift rates ``rte_rate`` [m/s] and ``roe_rate`` [rad/s], each [W]."""
        raise NotImplementedError

    def _rte_term(self, rte_rate: torch.Tensor) -> torch.Tensor:
        """Mean squared RTE drift rate; ``huber_delta`` [m/s] switches to Huber so heavy-tailed residuals
        get a bounded gradient."""
        if self.huber_delta is None:
            return rte_rate.square().mean()
        return F.huber_loss(rte_rate, torch.zeros_like(rte_rate), delta=self.huber_delta)

    def forward(
        self,
        model    : FoundationIMU,
        data     : IMUData_w_GT_Jagged,
        pred     : FoundationIMU_Output
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Graph-connected zero: see Denoise_Loss.
        applies = (data.labels.to(torch.int64) & int(self.LABEL)) != 0
        if not bool(applies.any()):
            return self._zero(pred), {}

        sub2full, sub_len, sub_offn = _restrict_to_applies(data, applies)
        if int(sub_len.max()) < 2:                     # preintegration needs >= one interval
            return self._zero(pred), {"loss": 0.0}

        device = sub2full.device
        n_sub  = sub_len.numel()
        run    = torch.repeat_interleave(torch.arange(n_sub, device=device), sub_len)

        # GT in the prediction's frame: translation rebased to each run's first row.
        gt = data.gt_pose.tensor()[sub2full].to(self.dtype)
        assert bool(torch.isfinite(gt).all()), \
            f"{type(self).__name__} needs finite gt_pose on every opted-in device"
        gt_rot = pp.SO3(gt[:, 3:7])
        gt_xyz = gt[:, :3] - gt[sub_offn[:-1], :3][run]

        xyz_hat, R_hat = self._integrate(data, pred, applies, sub2full, sub_offn, run, gt_rot)

        # Optimized: relative increments over each horizon's grid.
        t_ns    = data.time_ns[sub2full, 0]
        loss    = self._zero(pred)   # stays graph-connected even if every grid is empty
        metric: dict[str, float] = {}
        n_terms = 0
        for horizon, stride in zip(self.horizons, self.strides):
            i, j, _ = _grid_windows(data, sub2full, sub_offn, horizon, stride, self.unit)
            if i.numel() == 0:
                continue
            dt = (t_ns[j] - t_ns[i]).to(self.dtype) * 1e-9                    # [W] s

            # Steps in each trajectory's own start frame (evo RPE convention), so attitude error couples
            # into RTE as it does in the eval metrics.
            R_hat_i  = pp.SO3(R_hat.tensor()[i])
            gt_rot_i = pp.SO3(gt_rot.tensor()[i])
            step_hat = R_hat_i.Inv()  @ (xyz_hat[j] - xyz_hat[i])
            step_gt  = gt_rot_i.Inv() @ (gt_xyz[j]  - gt_xyz[i])
            rte      = (step_hat - step_gt).norm(dim=-1)                      # [W] m

            rel_hat = R_hat_i.Inv()  @ pp.SO3(R_hat.tensor()[j])
            rel_gt  = gt_rot_i.Inv() @ pp.SO3(gt_rot.tensor()[j])
            roe     = (rel_gt.Inv() @ rel_hat).Log().tensor().norm(dim=-1)    # [W] rad

            loss     = loss + self._horizon_term(rte / dt, roe / dt)
            n_terms += 1
            tag = f"{horizon:g}{self.unit}"
            metric[f"{self.METRIC_PREFIX}/rte_{tag}"] = float(rte.detach().square().mean().sqrt())
            metric[f"{self.METRIC_PREFIX}/roe_{tag}"] = float(roe.detach().square().mean().sqrt().rad2deg())
        loss = loss / max(n_terms, 1)

        # Metric-only: whole-window absolutes from the same integration.
        with torch.no_grad():
            ate = (xyz_hat - gt_xyz).norm(dim=-1)
            aoe = (gt_rot.Inv() @ R_hat).Log().tensor().norm(dim=-1)
            metric[f"{self.METRIC_PREFIX}/ate"] = float(ate.square().mean().sqrt())
            metric[f"{self.METRIC_PREFIX}/aoe"] = float(aoe.square().mean().sqrt().rad2deg())

        metric["loss"] = loss.item()
        return loss, metric


class Trajectory_Velocity_Loss(_Trajectory_LossBase):
    """Integrated-trajectory supervision for the body-velocity head.

    Attitude preintegrates the predicted denoised gyro from the GT initial rotation (detached by default,
    so this loss trains the velocity head alone); position integrates ``R̂ v̂_b`` with the trapezoid rule.
    ROE/AOE are metrics only. Requires finite ``gt_pose`` on opted-in devices.
    """

    LABEL         = Loss_Flags.Traj_Vel
    METRIC_PREFIX = "traj_vel"

    def __init__(
        self,
        horizons       : Sequence[float],
        strides        : Sequence[float] | None        = None,
        unit           : TokenUnit                     = "ms",
        integrator     : Literal["rk4", "euler"]       = "rk4",
        precision      : Literal["float32", "float64"] = "float32",
        huber_delta    : float | None                  = None,
        detach_rotation: bool                          = True,
    ):
        """``strides`` defaults to ``horizons``; ``detach_rotation=False`` lets the RTE gradient also reach
        the gyro head through the attitude chain."""
        super().__init__(horizons, strides, unit, integrator, precision, huber_delta)
        self.detach_rotation = detach_rotation

    def _zero(self, pred: FoundationIMU_Output) -> torch.Tensor:
        zero = pred.dec_body_vel.float().sum() * 0.0
        if not self.detach_rotation:
            # Keep the zero-grad set aligned with the real path so DDP sees the same parameters on every rank.
            zero = zero + pred.dec_correct_gyro.float().sum() * 0.0
        return zero

    def _integrate(self, data, pred, applies, sub2full, sub_offn, run, gt_rot):
        with torch.no_grad() if self.detach_rotation else nullcontext():
            gyro   = pred.dec_correct_gyro[sub2full].to(self.dtype)
            pack   = _channel_pack(data, applies, sub2full, sub_offn, torch.zeros_like(gyro), gyro)
            preint = self.preintegrate(pack, dtype=self.dtype)   # acc slot inert: only Dr is used
            Dr, *_ = _aligned_prefixes(preint, sub_offn)
        R_hat = pp.SO3(quaternion_multiply(gt_rot.tensor()[sub_offn[:-1]][run], Dr))
        v_b   = pred.dec_body_vel[sub2full].to(self.dtype)
        xyz   = integrate_trajectory(v_b, R_hat, data.time_ns[sub2full], sub_offn)
        return xyz, R_hat

    def _horizon_term(self, rte_rate: torch.Tensor, roe_rate: torch.Tensor) -> torch.Tensor:
        del roe_rate   # attitude is frozen context here, never a training target
        return self._rte_term(rte_rate)
