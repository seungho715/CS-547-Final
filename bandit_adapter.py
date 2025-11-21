# bandit_adapter.py
from __future__ import annotations
import math
import numpy as np
from typing import List, Optional, Dict


class SoftmaxUCBWeightBandit:

    def __init__(
        self,
        base_weights: List[float],
        eps: float = 0.2,
        temp: float = 0.25,
        c_ucb: float = 0.5,
        ema_beta: float = 0.6,
        wiggle_alpha: float = 0.0,  
        rng_seed: Optional[int] = None
    ):
        self.base = np.array(base_weights, dtype=np.float32)
        assert self.base.shape == (3,)
        assert 0.0 <= self.base[0] <= 1.0
        assert 0.0 <= self.base[2] <= 1.0
        assert 0.0 <= np.sum(self.base) <= 1.0 + 1e-6

        self.eps = float(eps)
        self.temp = float(temp)
        self.c_ucb = float(c_ucb)
        self.ema_beta = float(ema_beta)
        self.wiggle_alpha = float(wiggle_alpha)
        self.rng = np.random.default_rng(rng_seed)

        self.arms = self._make_arms()
        n = len(self.arms)
        self.counts = np.zeros(n, dtype=np.int32)
        self.means = np.zeros(n, dtype=np.float32)
        self.ema   = np.zeros(n, dtype=np.float32) 
        self.t     = 0 
        self.theta = self.base.copy()                 

    def pick_arm(self):
        self.t += 1
        scores = self._arm_scores_ucb_softmax()
        probs = self._softmax(scores / max(self.temp, 1e-6))
        i = int(self.rng.choice(len(self.arms), p=probs))
        arm = self._project(self.arms[i])
        self.theta = arm
        return i, self.theta

    def update(self, arm_idx: int, reward: float):
        r = float(reward)
        i = int(arm_idx)
        self.counts[i] += 1
        n = self.counts[i]
        self.means[i] += (r - self.means[i]) / max(n, 1)
        self.ema[i] = self.ema_beta * r + (1.0 - self.ema_beta) * self.ema[i]

    def _make_arms(self):
        w0, lyr0, a0 = float(self.base[0]), float(self.base[1]), float(self.base[2])
        dw_grid = np.array([-0.1, -0.05, 0.0, 0.05, 0.1], dtype=np.float32)
        da_grid = np.array([0.0], dtype=np.float32) if self.wiggle_alpha <= 1e-9 else np.array(
            [-self.wiggle_alpha, 0.0, self.wiggle_alpha], dtype=np.float32
        )

        arms = []
        for dw in dw_grid:
            wb = np.clip(w0 + dw, 0.0, 1.0)
            for da in da_grid:
                a = float(np.clip(a0 + da, 0.0, 1.0))
                wl = 1.0 - wb - a
                if wl < 0.0:
                    over = -(wl)
                    a = max(0.0, a - over)
                    wl = 1.0 - wb - a
                arm = np.array([wb, wl, a], dtype=np.float32)
                arms.append(self._project(arm))
        uniq = {tuple(np.round(a, 4)): a for a in arms}
        return np.stack(list(uniq.values()), axis=0)

    def _project(self, w: np.ndarray):
        w = w.copy()
        w = np.clip(w, 0.0, 1.0)
        s = w.sum()
        if s > 1.0:
            w = w / s
        diff = w - self.base
        nrm = np.linalg.norm(diff)
        if nrm > self.eps:
            w = self.base + diff * (self.eps / nrm)
            w = np.clip(w, 0.0, 1.0)
            s = w.sum()
            if s > 1.0:
                w = w / s
        return w.astype(np.float32)

    def _arm_scores_ucb_softmax(self):
        n = self.counts.astype(np.float32)
        mean = self.means
        ema = self.ema
        explo = self.c_ucb * np.sqrt(np.log(self.t + 1.0) / (n + 1.0))
        return 0.7 * ema + 0.3 * mean + explo

    @staticmethod
    def _softmax(x: np.ndarray):
        x = x - np.max(x)
        e = np.exp(x)
        z = e.sum()
        if not np.isfinite(z) or z <= 0:
            return np.ones_like(e) / len(e)
        return e / z
