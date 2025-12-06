# reward_policy.py

from typing import Dict, Any

# --- Policy Constants (Adjustable Hyperparameters) ---
# k: Threshold for an 'early skip' in seconds (30 seconds)
EARLY_SKIP_THRESHOLD_S = 30.0 
# lambda: Penalty weight for an early skip
EARLY_SKIP_PENALTY_LAMBDA = 0.5 


def calculate_reward(
    track_duration_s: float,
    play_time_s: float,
    is_skipped: bool,
    skip_latency_s: float = None,
) -> float:
    """
    Calculates the reward for a single track play session based on play duration and skip behavior.
    Formula: r = completion_ratio - lambda * 1[skip < k]
    """
    if track_duration_s <= 0:
        return 0.0

    # 1. Completion Ratio (Play Fraction)
    completion_ratio = min(1.0, play_time_s / track_duration_s)

    # 2. Skip Penalty Logic
    penalty = 0.0
    if is_skipped:
        latency = skip_latency_s if skip_latency_s is not None else play_time_s
        
        # Apply penalty if skip occurs before the threshold k
        if latency < EARLY_SKIP_THRESHOLD_S:
            penalty = EARLY_SKIP_PENALTY_LAMBDA

    # 3. Final Reward: Reward = Completion Ratio - Penalty
    final_reward = completion_ratio - penalty
    return final_reward