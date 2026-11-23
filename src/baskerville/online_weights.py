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

Use: ``params["train"]["loss_weights"]["hook"] = {"class":
"baskerville.online_weights:GradientAlignmentHook", "args": {...}}`` with a class
table and ``meta_valid``. eta = 0 leaves the weights (and training) unchanged.
"""

from __future__ import annotations

import time

import torch

from baskerville import loss_weights


class GradientAlignmentHook:
    def __init__(
        self,
        eta: float = 0.05,
        every: int = 10,
        eps: float = 1e-3,
        warmup_steps: int = 0,
        lo: float = 0.2,
        hi: float = 5.0,
        ema: float = 0.9,
        precondition: bool = False,
        adam_eps: float = 1e-8,
        seed: int = 0,
        di: int = 0,
    ):
        self.eta, self.every, self.eps = eta, every, eps
        self.warmup_steps, self.lo, self.hi = warmup_steps, lo, hi
        self.ema, self.precondition, self.adam_eps = ema, precondition, adam_eps
        self.seed, self.di = seed, di
        self.scale = None  # EMA of rms(a)
        self.updates = 0

    def state_dict(self):
        return {"scale": self.scale, "updates": self.updates}

    def load_state_dict(self, state):
        self.scale, self.updates = state["scale"], state["updates"]

    def val_direction(self, trainer, step):
        """Unit direction of grad_theta L_val on the next meta batch, and |g_val|."""
        meta = trainer.next_meta_batch()
        params = trainer.trainable_params()
        yv = trainer.forward(meta, rng_seed=self.seed * 1_000_003 + step, grad=True)
        l_val = trainer.val_objective(yv, meta)
        # backward, not autograd.grad: reentrant gradient checkpointing needs it
        for p in params:
            p.grad = None
        l_val.backward()
        grads = [
            torch.zeros_like(p) if p.grad is None else p.grad.detach().float().clone()
            for p in params
        ]
        for p in params:
            p.grad = None
        if self.precondition:
            moments = trainer.optimizer_second_moments()
            grads = [
                g if v is None else g / (v.float().sqrt() + self.adam_eps)
                for g, v in zip(grads, moments)
            ]
        norm = torch.sqrt(sum((g * g).sum() for g in grads))
        return [g / norm for g in grads], float(norm), float(l_val.detach())

    def alignment(self, trainer, batch, direction, step, weights):
        """a = d/dW of the central difference of L_train along ``direction``."""
        seed = self.seed * 1_000_003 + step + 500_009
        trainer.perturb_params(direction, self.eps)
        y_plus = trainer.forward(batch, rng_seed=seed).detach()
        trainer.restore_params()
        trainer.perturb_params(direction, -self.eps)
        y_minus = trainer.forward(batch, rng_seed=seed).detach()
        trainer.restore_params()
        W = weights.detach().clone().requires_grad_(True)
        diff = trainer.loss_from_pred(
            y_plus, batch, W, self.di
        ) - trainer.loss_from_pred(y_minus, batch, W, self.di)
        (a,) = torch.autograd.grad(diff / (2 * self.eps), W)
        return a

    def before_step(self, trainer, batch, di):
        step = trainer._step
        if di != self.di or step < self.warmup_steps or step % self.every:
            return
        t0 = time.time()
        W = trainer.class_weights[self.di]
        direction, g_norm, l_val = self.val_direction(trainer, step)
        a = self.alignment(trainer, batch, direction, step, W)
        rms = float(a.pow(2).mean().sqrt())
        self.scale = (
            rms if self.scale is None else self.ema * self.scale + (1 - self.ema) * rms
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
            "L_val": l_val,
            "g_val_norm": g_norm,
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
        row.update(
            {
                f"a[{c},{g}]": float(a[c, g])
                for c in range(a.shape[0])
                for g in range(a.shape[1])
            }
        )
        trainer.log_hook_update(row)
