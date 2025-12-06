from __future__ import annotations
from typing import Dict, Any, List, Tuple
import random
import numpy as np

from reward_policy import calculate_reward, EARLY_SKIP_THRESHOLD_S, EARLY_SKIP_PENALTY_LAMBDA

from feature_store import FeatureStore
from candidate_gen import generate_candidates
from scorer import score_track
from bandit_adapter import SoftmaxUCBWeightBandit
from flask import Flask
app = Flask(__name__)
@app.route('/')
def home():
    return "Hello, this is the music recommendation system!"

# Minimal config
RANDOM_SEED = 42
TOPK_CANDIDATES = 300
MMR_LAMBDA = 0.7
DELTA_BPM_DEFAULT = 6
EPS_GREEDY_ON_SKIP = 0.15
SESSION_PENALTY = 0.15
BANDIT_UPDATE_EVERY = 3
HISTORY_N_FOR_QUERY = 3

# --- Simulation Constant ---
TRACK_DURATION_S = 180.0

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Init data & components
fs = FeatureStore("dataset/artifacts")
track_meta_by_index = {i: fs.get_track_data(fs.ids[i]) for i in range(len(fs.ids))}

query: Dict[str, Any] = {
    "bpm": 128.0,
    "delta": DELTA_BPM_DEFAULT,
    "lyr_emb": None,
    "aud_emb": None,
}

# reinit bandit on slider change; replace with bandit.set_base(...) later
slider_w, alpha = 0.7, 0.0
base = [slider_w, max(0.0, 1.0 - slider_w - alpha), alpha]
bandit = SoftmaxUCBWeightBandit(base, eps=0.2, rng_seed=RANDOM_SEED)

# Helpers
def mean_unit(vecs: List[np.ndarray] | List[None]) -> np.ndarray | None:
    vecs = [v for v in vecs if v is not None]
    if not vecs:
        return None
    v = np.stack(vecs).mean(axis=0)
    n = np.linalg.norm(v)
    return v / (n + 1e-12)

# --- Build a query lyric embedding from 3 random seeds that have lyrics ---
eligible = [i for i in range(len(fs.ids)) if track_meta_by_index[i].get("lyr_emb") is not None]
if len(eligible) >= 3:
    seed_indices = random.sample(eligible, 3)
    seed_lyr_vecs = [track_meta_by_index[i]["lyr_emb"] for i in seed_indices]
    query["lyr_emb"] = mean_unit(seed_lyr_vecs)

# 1) pick dynamic weights
arm_idx, theta = bandit.pick_arm()

# if lyrics weight > bpm, prefer tracks that actually have lyrics
query["lyric_pref_filter"] = (theta[1] > theta[0])

# 2) generate candidates 
hits = generate_candidates(
    fs,
    query_profile=query,
    track_meta=track_meta_by_index,
    k_ann=300,
    delta=query["delta"],
    use_mmr=True,
    lambda_mmr=0.7
)

# If lyric_pref_filter is set, remove items without lyrics
if query.get("lyric_pref_filter"):
    filtered = [(i, s) for (i, s) in hits if track_meta_by_index[i].get("lyr_emb") is not None]
    hits = filtered or hits

# 3) score candidates with theta = [wbpm, wlyrics, waudio]
scored: List[Tuple[int, float, Dict]] = []
for idx, ann_score in hits:
    track = track_meta_by_index[idx]
    final_score, parts = score_track(track, query, theta)
    scored.append((idx, final_score, parts))

ranked = sorted(scored, key=lambda x: x[1], reverse=True)

# --- Simulate session ---

top_i, top_score, parts = ranked[0]
print(f"Top Recommended Track Index: {top_i} (Score: {top_score:.4f})")
print(f"Weights used: [BPM={theta[0]:.3f}, Lyrics={theta[1]:.3f}, Audio={theta[2]:.3f}]")
print(f"Score parts: Sbpm={parts['Sbpm']:.3f} Slyrics={parts['Slyrics']:.3f} Saudio={parts['Saudio']:.3f}")

# 4) Simulate user event and calculate reward (r = completion_ratio - lambda * 1[skip < k])
simulated_skip = (theta[1] > theta[0]) and (parts["Slyrics"] < 0.40)
if simulated_skip:
    # --- BAD recommendation (Early Skip) ---
    is_skipped = True
    completion_ratio, skip_latency_s = 0.05, 3.0
    play_time_s = completion_ratio * TRACK_DURATION_S
else:
    # --- GOOD recommendation (Full Listen) ---
    is_skipped = False
    completion_ratio = 0.90
    play_time_s = completion_ratio * TRACK_DURATION_S
    skip_latency_s = play_time_s # No skip, so latency is end of play time

r = calculate_reward(TRACK_DURATION_S, play_time_s, is_skipped, skip_latency_s)

# 5) Bandit Update and Log History

# Update the bandit's statistics with the reward
bandit.update(arm_idx, r) 
print(f"[Bandit Update] Arm {arm_idx} updated with Reward: {r:.4f}")

# Log History
history = [] # Initialize history list for this demonstration
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
    per_track_penalty = {} # placeholder for the original file's logic
    per_track_penalty[top_i] = per_track_penalty.get(top_i, 0.0) + SESSION_PENALTY
    
else:
    accepted_indices = [] # placeholder for the original file's logic
    accepted_indices.append(top_i)



# reinit bandit on slider change; replace with bandit.set_base(...) later
def on_slider_change(new_w: float, a: float = 0.0):
    global bandit
    new_base = [new_w, max(0.0, 1.0 - new_w - a), a]
    # TODO (FRONTEND): call this when the user moves the slider (debounced)
    bandit = SoftmaxUCBWeightBandit(new_base, eps=0.2, rng_seed=RANDOM_SEED)

# seed query lyr-embedding from initial user-provided songs (ids)
def seed_from_frontend(track_ids: List[str]):
    # TODO (FRONTEND): call once at session start with 3–5 seed track_ids
    idxs = []
    for tid in track_ids:
        where = np.where(fs.ids == tid)[0]
        if len(where):
            idxs.append(int(where[0]))
    accepted_indices.extend(idxs)
    refresh_query_from_last_N()

# Ranking utilities
per_track_penalty: Dict[int, float] = {}
history: List[Dict[str, Any]] = []
accepted_indices: List[int] = []
pending_rewards: List[Tuple[int, float]] = []

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

def refresh_query_from_last_N():
    last_N = accepted_indices[-HISTORY_N_FOR_QUERY:]
    if not last_N:
        return
    vecs = [track_meta_by_index[i].get("lyr_emb") for i in last_N]
    query["lyr_emb"] = mean_unit(vecs)

# Bootstrap query embedding
eligible = [i for i in range(len(fs.ids)) if track_meta_by_index[i].get("lyr_emb") is not None]
if len(eligible) >= HISTORY_N_FOR_QUERY and query["lyr_emb"] is None:
    seeds = random.sample(eligible, HISTORY_N_FOR_QUERY)
    seed_vecs = [track_meta_by_index[i]["lyr_emb"] for i in seeds]
    query["lyr_emb"] = mean_unit(seed_vecs)

# Session loop (demo)
plays_to_simulate = 10

for step in range(plays_to_simulate):
    arm_idx, theta = bandit.pick_arm()
    ranked = rank_once(theta)
    top_i, top_score, parts = ranked[0]
    print(f"[serve] idx={top_i} score={top_score:.4f} w_used={tuple(round(x,2) for x in parts['w_used'])} "
          f"Sbpm={parts['Sbpm']:.3f} Slyrics={parts['Slyrics']:.3f} Saudio={parts['Saudio']:.3f}")

    # will need to replace it with FRONTEND-provided data
    simulated_skip = (theta[1] > theta[0]) and (parts["Slyrics"] < 0.40)
    if simulated_skip:
        completion_ratio, skip_latency_s = 0.05, 3.0
    else:
        completion_ratio, skip_latency_s = 0.90, 999.0

    r = reward_from_event(completion_ratio, skip_latency_s)
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
        # can immediately attempt to play alt_idx in the UI if desired
    else:
        accepted_indices.append(top_i)
        pending_rewards.append((arm_idx, r))

    if len(pending_rewards) >= BANDIT_UPDATE_EVERY:
        mean_r = float(np.mean([x[1] for x in pending_rewards]))
        bandit.update(pending_rewards[-1][0], mean_r)
        pending_rewards.clear()
        refresh_query_from_last_N()

print(f"\nSession done. plays={len(history)}, accepted={len(accepted_indices)}")
