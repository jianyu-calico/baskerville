"""Online class-weight updates by validation-gradient alignment (Ren et al. 2018).

First order, with a finite-difference look-ahead (see the online-reweighting-prep
method design). Every ``every`` steps, before the training step on a batch:

1. g_val = grad_theta L_val on the next meta-validation batch (``trainer.val_objective``,
   i.e. L_pmn + meta_valid.dev_lambda * L_dev), normalized to a unit direction
   (optionally preconditioned by Adam's 1 / sqrt(exp_avg_sq)).
2. Two no-grad forwards of the training batch at theta +- eps * g, same dropout masks.
3. a[c, g] = d/dW of (L_train(theta + eps g, W) - L_train(theta - eps g, W)) / (2 eps):
   how much raising each class x group weight aligns the training step with the
   validation gradient. Only the loss is differentiated (W enters after the model).
4. W <- project(W * exp(eta * a / EMA(rms(a)))): clip to [lo, hi] with mean 1 over
   training bins per group (``loss_weights.project_weights``), written in place.

Precision: the finite-difference forwards run in ``fd_dtype``; "auto" (default) is
float32 when training uses bf16/fp16 autocast, because bf16 outputs (relative ulp
~4e-3) make y(theta + eps g) - y(theta - eps g) resolution-limited at medium size.
The step is ``eps`` (absolute) or ``eps_rel * ||theta||`` (recomputed each update).
Each update also checks the finite difference against the exact gradient: the FD
derivative of L_val along the direction must match g_val . direction (``check_tol``
relative; ``check_action`` "warn" or "abort"). The check probes along g_val's own
direction on the meta batch, so it is necessary, not sufficient, for an accurate
alignment on the training batch; with a bf16 g_val the ratio can drift. Keep
``check_action="warn"`` until it is calibrated at the model size used. Non-finite
updates are skipped.
``fd_scan = {rel, ref_rel, every_updates}`` logs, on every N-th update, the alignment
at several steps in fp32 and in the training precision (fd_scan.tsv), without
affecting training.

Use: ``params["train"]["loss_weights"]["hook"] = {"class":
"baskerville.online_weights:GradientAlignmentHook", "args": {...}}`` with a class
table and ``meta_valid``. eta = 0 leaves the weights (and training) unchanged.
"""

from __future__ import annotations

import math
import time
import warnings

import torch

from baskerville import loss_weights

DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def theta_norm(trainer) -> float:
    """||theta|| over the trainable parameters."""
    with torch.no_grad():
        return float(
            torch.sqrt(sum(p.float().pow(2).sum() for p in trainer.trainable_params()))
        )


class GradientAlignmentHook:
    def __init__(
        self,
        eta: float = 0.05,
        every: int = 10,
        eps: float = 1e-3,
        eps_rel: float | None = None,
        fd_dtype: str = "auto",
        fd_scan: dict | None = None,
        check: bool = True,
        check_tol: float = 0.2,
        check_action: str = "warn",
        warmup_steps: int = 0,
        lo: float = 0.2,
        hi: float = 5.0,
        ema: float = 0.9,
        precondition: bool = False,
        adam_eps: float = 1e-8,
        seed: int = 0,
        di: int = 0,
    ):
        if fd_dtype != "auto" and fd_dtype not in DTYPES:
            raise ValueError(f"fd_dtype must be 'auto' or one of {list(DTYPES)}")
        if check_action not in ("warn", "abort"):
            raise ValueError("check_action must be 'warn' or 'abort'")
        self.eta, self.every, self.eps, self.eps_rel = eta, every, eps, eps_rel
        self.fd_dtype, self.fd_scan = fd_dtype, fd_scan or None
        self.check, self.check_tol, self.check_action = check, check_tol, check_action
        self.warmup_steps, self.lo, self.hi = warmup_steps, lo, hi
        self.ema, self.precondition, self.adam_eps = ema, precondition, adam_eps
        self.seed, self.di = seed, di
        self.scale = None  # EMA of rms(a)
        self.updates = 0
        self.skipped = 0
        self._warned_moments = False

    def state_dict(self):
        return {"scale": self.scale, "updates": self.updates, "skipped": self.skipped}

    def load_state_dict(self, state):
        self.scale, self.updates = state["scale"], state["updates"]
        self.skipped = state.get("skipped", 0)

    # seeds: the val forward and the two FD forwards each get their own fixed seed per
    # step; +eps and -eps share one (identical dropout masks)
    def val_seed(self, step):
        return self.seed * 1_000_003 + step

    def fd_seed(self, step):
        return self.seed * 1_000_003 + step + 500_009

    def resolved_fd_dtype(self, trainer):
        if self.fd_dtype != "auto":
            return DTYPES[self.fd_dtype]
        return (
            torch.float32
            if trainer.mix_dtype in (torch.bfloat16, torch.float16)
            else trainer.mix_dtype
        )

    def val_direction(self, trainer, step, meta=None):
        """Unit direction of (preconditioned) grad_theta L_val on a meta batch.

        Returns (direction, |g_val|, L_val, g_val . direction, meta batch); the
        direction is None when the gradient is zero or not finite.
        """
        meta = trainer.next_meta_batch() if meta is None else meta
        params = trainer.trainable_params()
        for p in params:
            p.grad = None
        # forward and backward inside frozen_norm_stats: checkpointed blocks recompute
        # their forward during backward. backward (not autograd.grad) for reentrant
        # gradient checkpointing.
        with trainer.frozen_norm_stats():
            yv = trainer.forward(meta, rng_seed=self.val_seed(step), grad=True)
            l_val = trainer.val_objective(yv, meta)
            l_val.backward()
        grads = [
            torch.zeros_like(p) if p.grad is None else p.grad.detach().float().clone()
            for p in params
        ]
        for p in params:
            p.grad = None
        g_norm = float(torch.sqrt(sum((g * g).sum() for g in grads)))
        direction = grads
        if self.precondition:
            moments = trainer.optimizer_second_moments()
            if any(v is None for v in moments) and not self._warned_moments:
                warnings.warn(
                    "precondition=True but some exp_avg_sq are missing; those use plain gradients"
                )
                self._warned_moments = True
            direction = [
                g if v is None else g / (v.float().sqrt() + self.adam_eps)
                for g, v in zip(grads, moments)
            ]
        d_norm = float(torch.sqrt(sum((d * d).sum() for d in direction)))
        if not (math.isfinite(d_norm) and d_norm > 0):
            return None, g_norm, float(l_val.detach()), float("nan"), meta
        direction = [d / d_norm for d in direction]
        g_dot_d = float(sum((g * d).sum() for g, d in zip(grads, direction)))
        return direction, g_norm, float(l_val.detach()), g_dot_d, meta

    def step_size(self, trainer):
        """eps: absolute, or eps_rel x ||theta|| (recomputed at every update)."""
        if self.eps_rel is None:
            return self.eps
        return self.eps_rel * theta_norm(trainer)

    def alignment(self, trainer, batch, direction, step, weights, eps=None, dtype=None):
        """a = d/dW of the central difference of L_train along ``direction``."""
        eps = self.eps if eps is None else eps
        dtype = self.resolved_fd_dtype(trainer) if dtype is None else dtype
        seed = self.fd_seed(step)
        trainer.perturb_params(direction, eps)
        y_plus = trainer.forward(batch, rng_seed=seed, dtype=dtype).detach()
        trainer.restore_params()
        trainer.perturb_params(direction, -eps)
        y_minus = trainer.forward(batch, rng_seed=seed, dtype=dtype).detach()
        trainer.restore_params()
        W = weights.detach().clone().requires_grad_(True)
        diff = trainer.loss_from_pred(
            y_plus, batch, W, self.di, dtype=dtype
        ) - trainer.loss_from_pred(y_minus, batch, W, self.di, dtype=dtype)
        (a,) = torch.autograd.grad(diff / (2 * eps), W)
        return a

    def fd_check(self, trainer, meta, direction, step, eps, g_dot_d):
        """Ratio of the FD derivative of L_val along the direction to g_val . direction
        (exact); ~1 when the finite difference is resolved. Same dropout masks as the
        gradient."""
        dtype = self.resolved_fd_dtype(trainer)
        vals = []
        for sign in (1, -1):
            trainer.perturb_params(direction, sign * eps)
            with torch.no_grad():
                y = trainer.forward(meta, rng_seed=self.val_seed(step), dtype=dtype)
                vals.append(float(trainer.val_objective(y, meta, dtype=dtype)))
            trainer.restore_params()
        fd = (vals[0] - vals[1]) / (2 * eps)
        ratio = fd / g_dot_d if g_dot_d else float("nan")
        if not (abs(ratio - 1) <= self.check_tol):
            msg = (
                f"online weights, step {step}: FD derivative of L_val {fd:.4g} vs exact "
                f"{g_dot_d:.4g} (ratio {ratio:.3g}); the finite difference isn't resolved "
                "(fd_dtype / eps / eps_rel)"
            )
            if self.check_action == "abort":
                raise RuntimeError(msg)
            warnings.warn(msg)
        return ratio

    def run_fd_scan(self, trainer, batch, direction, step, W):
        """Diagnostic: the alignment in fp32 (autocast off) at each ref_rel and in the
        training precision at each rel, on the same batch and direction; logged to
        fd_scan.tsv with correlation and sign agreement vs the first fp32 reference.
        Weights, parameters and RNG are left as they were."""
        norm = theta_norm(trainer)
        runs = []
        for rel in self.fd_scan.get("ref_rel", [1e-5, 1e-4]):
            runs.append(
                (
                    "float32",
                    rel,
                    self.alignment(
                        trainer, batch, direction, step, W, rel * norm, torch.float32
                    ),
                )
            )
        train_dtype = trainer.mix_dtype
        precision = str(train_dtype).replace("torch.", "")
        for rel in self.fd_scan.get("rel", [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]):
            runs.append(
                (
                    precision,
                    rel,
                    self.alignment(
                        trainer, batch, direction, step, W, rel * norm, train_dtype
                    ),
                )
            )
        ref = runs[0][2].flatten().double()
        for kind, rel, a in runs:
            x = a.flatten().double()
            ok = x.std() > 0 and ref.std() > 0
            corr = (
                float(torch.corrcoef(torch.stack([x, ref]))[0, 1])
                if ok
                else float("nan")
            )
            row = {
                "update": self.updates,
                "step": step,
                "precision": kind,
                "rel": rel,
                "eps": rel * norm,
                "theta_norm": norm,
                "corr_vs_ref": corr,
                "sign_agree_vs_ref": float(
                    (torch.sign(x) == torch.sign(ref)).double().mean()
                ),
            }
            row.update(
                {
                    f"a[{c},{g}]": float(a[c, g])
                    for c in range(a.shape[0])
                    for g in range(a.shape[1])
                }
            )
            trainer.log_hook_update(row, "fd_scan.tsv")

    def before_step(self, trainer, batch, di):
        step = trainer._step
        if di != self.di or step < self.warmup_steps or step % self.every:
            return
        t0 = time.time()
        W = trainer.class_weights[self.di]
        direction, g_norm, l_val, g_dot_d, meta = self.val_direction(trainer, step)
        eps = self.step_size(trainer)
        a = ratio = None
        if direction is not None:
            a = self.alignment(trainer, batch, direction, step, W, eps)
            if self.check:
                ratio = self.fd_check(trainer, meta, direction, step, eps, g_dot_d)
            if (
                self.fd_scan
                and self.updates % int(self.fd_scan.get("every_updates", 1)) == 0
            ):
                self.run_fd_scan(trainer, batch, direction, step, W)
        rms = float(a.pow(2).mean().sqrt()) if a is not None else float("nan")
        skipped = a is None or not torch.isfinite(a).all() or not math.isfinite(rms)
        if skipped:
            self.skipped += 1
            warnings.warn(
                f"online weights, step {step}: non-finite or zero gradient/alignment; update skipped"
            )
        else:
            self.scale = (
                rms
                if self.scale is None
                else self.ema * self.scale + (1 - self.ema) * rms
            )
            if self.eta != 0 and self.scale > 0:
                new = W * torch.exp(self.eta * a / self.scale)
                new = loss_weights.project_weights(
                    new, trainer.class_bin_counts[self.di], self.lo, self.hi
                )
                with torch.no_grad():
                    W.copy_(new)
        self.updates += 1
        row = {
            "step": step,
            "skipped": int(skipped),
            "L_val": l_val,
            "g_val_norm": g_norm,
            "eps": eps,
            "fd_check_ratio": float("nan") if ratio is None else ratio,
            "a_rms": rms,
            "seconds": round(time.time() - t0, 4),
        }
        row.update(
            {
                f"w[{c},{g}]": float(W[c, g])
                for c in range(W.shape[0])
                for g in range(W.shape[1])
            }
        )
        if a is not None:
            row.update(
                {
                    f"a[{c},{g}]": float(a[c, g])
                    for c in range(a.shape[0])
                    for g in range(a.shape[1])
                }
            )
        else:
            row.update(
                {
                    f"a[{c},{g}]": float("nan")
                    for c in range(W.shape[0])
                    for g in range(W.shape[1])
                }
            )
        trainer.log_hook_update(row)
