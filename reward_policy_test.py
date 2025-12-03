import numpy as np
from bandit_adapter import SoftmaxUCBWeightBandit 
from typing import Dict, Any

EARLY_SKIP_THRESHOLD_MS = 30000  
EARLY_SKIP_PENALTY_LAMBDA = 0.5   


def calculate_reward(
    track_duration_ms: float,
    play_time_ms: float,
    is_skipped: bool,
    skip_latency_ms: float = None,
) -> float:
    
    if track_duration_ms <= 0:
        return 0.0

    completion_ratio = min(1.0, play_time_ms / track_duration_ms)

    penalty = 0.0
    if is_skipped:
        latency = skip_latency_ms if skip_latency_ms is not None else play_time_ms
        
        if latency < EARLY_SKIP_THRESHOLD_MS:
            penalty = EARLY_SKIP_PENALTY_LAMBDA

    final_reward = completion_ratio - penalty
    return final_reward


# --- Compatibility and Testing Sketch ---

def test_bandit_compatibility():

    slider_w, alpha = 0.7, 0.0
    base = [slider_w, max(0.0, 1.0 - slider_w - alpha), alpha]
    bandit = SoftmaxUCBWeightBandit(base, eps=0.2, rng_seed=42)

    DURATION_3MIN = 180000.0 

    # --- Session 1: Late Skip (Good Performance) ---
    arm_idx_1, theta_1 = bandit.pick_arm()
    play_time_3 = 60000.0
    reward_3 = calculate_reward(DURATION_3MIN, play_time_3, is_skipped=True, skip_latency_ms=play_time_3)
    
    bandit.update(arm_idx_1, reward_3)
    
    # --- Session 2: Early Skip (Bad Performance) ---
    arm_idx_2, theta_2 = bandit.pick_arm()
    play_time_2 = 5000.0
    reward_2 = calculate_reward(DURATION_3MIN, play_time_2, is_skipped=True, skip_latency_ms=play_time_2)
    
    bandit.update(arm_idx_2, reward_2)
    
    
if __name__ == '__main__':
    test_bandit_compatibility()
