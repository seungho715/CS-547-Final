from __future__ import annotations
from typing import Dict, Any, List, Tuple
import random
import numpy as np
from reward_policy import calculate_reward
from feature_store import FeatureStore
from candidate_gen import generate_candidates
from scorer import score_track
from bandit_adapter import SoftmaxUCBWeightBandit
from flask import Flask
app = Flask(__name__)
@app.route('/')
def home():
    return "Hello, this is the music recommendation system!"

# Config
RANDOM_SEED = 42
TOPK_CANDIDATES = 300
MMR_LAMBDA = 0.7
DELTA_BPM_DEFAULT = 6
EPS_GREEDY_ON_SKIP = 0.15
SESSION_PENALTY = 0.15
BANDIT_UPDATE_EVERY = 3
HISTORY_N_FOR_QUERY = 3
TRACK_DURATION_S = 180.0 

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

per_track_penalty: Dict[int, float] = {}
history: List[Dict[str, Any]] = []
accepted_indices: List[int] = []
pending_rewards: List[Tuple[int, float]] = []

fs = FeatureStore("dataset/artifacts")
track_meta_by_index = {i: fs.get_track_data(fs.ids[i]) for i in range(len(fs.ids))}

query: Dict[str, Any] = {
    "bpm": 128.0,
    "delta": DELTA_BPM_DEFAULT,
    "lyr_emb": None,
    "aud_emb": None,
}

slider_w, alpha = 0.7, 0.0
base = [slider_w, max(0.0, 1.0 - slider_w - alpha), alpha]
bandit = SoftmaxUCBWeightBandit(base, eps=0.2, rng_seed=RANDOM_SEED)

def mean_unit(vecs: List[np.ndarray] | List[None]) -> np.ndarray | None:
    vecs = [v for v in vecs if v is not None]
    if not vecs:
        return None
    v = np.stack(vecs).mean(axis=0)
    n = np.linalg.norm(v)
    return v / (n + 1e-12)

def on_slider_change(new_w: float, a: float = 0.0):
    """FRONTEND will call this when user moves the slider"""
    global bandit
    new_base = [new_w, max(0.0, 1.0 - new_w - a), a]
    bandit = SoftmaxUCBWeightBandit(new_base, eps=0.2, rng_seed=RANDOM_SEED)

def refresh_query_from_last_N():
    last_N = accepted_indices[-HISTORY_N_FOR_QUERY:]
    if not last_N:
        return
    vecs = [track_meta_by_index[i].get("lyr_emb") for i in last_N]
    query["lyr_emb"] = mean_unit(vecs)

def rank_once(theta: np.ndarray) -> List[Tuple[int, float, Dict[str, Any]]]:
    hits = generate_candidates(
        fs,
        query_profile=query,
        track_meta=track_meta_by_index,
        k_ann=TOPK_CANDIDATES,
        delta=query.get("delta", DELTA_BPM_DEFAULT),
        use_mmr=True,
        lambda_mmr=MMR_LAMBDA
    )
    # Prefer tracks with lyrics if lyrics weight > BPM
    if theta[1] > theta[0]:
        filtered = [(i, s) for (i, s) in hits if track_meta_by_index[i].get("lyr_emb") is not None]
        hits = filtered or hits

    scored: List[Tuple[int, float, Dict[str, Any]]] = []
    for i, _ in hits:
        s, parts = score_track(track_meta_by_index[i], query, theta)
        s -= per_track_penalty.get(i, 0.0)
        scored.append((i, s, parts))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored

def choose_next(ranked: List[Tuple[int, float, Dict[str, Any]]], on_skip: bool) -> int:
    if not ranked:
        raise RuntimeError("No candidates to choose from.")
    if on_skip and random.random() < EPS_GREEDY_ON_SKIP and len(ranked) >= 10:
        return ranked[random.randint(1, 9)][0]
    return ranked[0][0]

eligible = [i for i in range(len(fs.ids)) if track_meta_by_index[i].get("lyr_emb") is not None]
if len(eligible) >= HISTORY_N_FOR_QUERY and query["lyr_emb"] is None:
    seeds = random.sample(eligible, HISTORY_N_FOR_QUERY)
    seed_vecs = [track_meta_by_index[i]["lyr_emb"] for i in seeds]
    query["lyr_emb"] = mean_unit(seed_vecs)

plays_to_simulate = 10

for step in range(plays_to_simulate):
    arm_idx, theta = bandit.pick_arm()
    query["lyric_pref_filter"] = (theta[1] > theta[0])

    ranked = rank_once(theta)
    top_i, top_score, parts = ranked[0]
    print(f"[serve] idx={top_i} score={top_score:.4f} w_used={tuple(round(x,2) for x in parts['w_used'])} "
          f"Sbpm={parts['Sbpm']:.3f} Slyrics={parts['Slyrics']:.3f} Saudio={parts['Saudio']:.3f}")
    
    # added simulated track_duration
    meta = track_meta_by_index[top_i]
    duration = meta.get("duration_ms")
    track_duration_s = float(duration) / 1000.0 if duration is not None else 180.0

    simulated_skip = (theta[1] > theta[0]) and (parts["Slyrics"] < 0.40)
    if simulated_skip:
        completion_ratio = 0.05
        skip_latency_s   = 3.0
        is_skipped       = True
    else:
        completion_ratio = 0.90
        skip_latency_s   = track_duration_s
        is_skipped       = False

    play_time_s = completion_ratio * track_duration_s
    
    r = calculate_reward(track_duration_s, play_time_s, is_skipped, skip_latency_s)

    history.append({
        "track_index": int(top_i),
        "theta": [float(x) for x in theta],
        "Sbpm": float(parts["Sbpm"]),
        "Slyrics": float(parts["Slyrics"]),
        "Saudio": float(parts["Saudio"]),
        "completion_ratio": float(completion_ratio),
        "skip_latency_s": float(skip_latency_s),
        "reward": float(r),
    })

    if simulated_skip:
        per_track_penalty[top_i] = per_track_penalty.get(top_i, 0.0) + SESSION_PENALTY
        alt_idx = choose_next(ranked, on_skip=True)
        print(f"[skip] penalized idx={top_i}; trying alt idx={alt_idx}")
    else:
        accepted_indices.append(top_i)
        pending_rewards.append((arm_idx, r))

    if len(pending_rewards) >= BANDIT_UPDATE_EVERY:
        mean_r = float(np.mean([x[1] for x in pending_rewards]))
        bandit.update(pending_rewards[-1][0], mean_r)
        pending_rewards.clear()
        refresh_query_from_last_N()

print(f"\nSession done. plays={len(history)}, accepted={len(accepted_indices)}")
