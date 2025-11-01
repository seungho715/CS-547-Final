import numpy as np

class SoftmaxBandit:
    def __init__(self, base_weights, epsilon=0.2, temp=0.2):
        self.base = np.array(base_weights, dtype="float32")
        self.theta = self.base.copy()
        self.eps = float(epsilon)
        self.temp = float(temp)
        self.arms = self._make_arms()

    def _make_arms(self):
        deltas = [-0.1, 0.0, 0.1]
        arms = []
        for dw in deltas:
            w = np.clip(self.base[0] + dw, 0.0, 1.0)
            alpha = np.clip(self.base[2], 0.0, 0.2)
            arms.append(np.array([w, 1-w, alpha], dtype="float32"))
        return arms

    def pick_arm(self, rewards_history=None):
        if not rewards_history:
            probs = np.ones(len(self.arms))/len(self.arms)
        else:
            means = np.array([np.mean(r) if r else 0.0 for r in rewards_history], dtype="float32")
            logits = means / max(self.temp, 1e-6)
            e = np.exp(logits - logits.max())
            probs = e / e.sum()
        i = np.random.choice(len(self.arms), p=probs)
        arm = self.arms[i]
        diff = arm - self.base
        if np.linalg.norm(diff) > self.eps:
            arm = self.base + diff * (self.eps / np.linalg.norm(diff))
        self.theta = arm
        return i, self.theta
