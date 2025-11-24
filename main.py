from feature_store import FeatureStore
from candidate_gen import generate_candidates
from scorer import score_track
from bandit_adapter import SoftmaxUCBWeightBandit
from typing import Dict, Any, List, Tuple
import numpy as np
import random

# --- Initialization ---
fs = FeatureStore("dataset/artifacts")

# int-index → metadata (tempo + embeddings) for BPM filter/MMR/scoring
track_meta_by_index = {i: fs.get_track_data(fs.ids[i]) for i in range(len(fs.ids))}

# Query profile (add fields as needed)
query: Dict[str, Any] = {"bpm": 128.0, "delta": 6, "lyr_emb": None, "aud_emb": None}

# Bandit base weights from slider; keep alpha 0.0 until you have real audio embeddings
slider_w, alpha = 0.7, 0.0
base = [slider_w, max(0.0, 1.0 - slider_w - alpha), alpha]
bandit = SoftmaxUCBWeightBandit(base, eps=0.2, rng_seed=42)

def mean_unit(vecs):
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

# (optional) if lyrics weight > bpm, prefer tracks that actually have lyrics
query["lyric_pref_filter"] = (theta[1] > theta[0])

# 2) generate candidates (use MMR to diversify)
hits = generate_candidates(
    fs,
    query_profile=query,
    track_meta=track_meta_by_index,
    k_ann=300,
    delta=query["delta"],
    use_mmr=True,
    lambda_mmr=0.7
)

# If lyric_pref_filter is set, remove items without lyrics (fallback to hits if it empties)
if query.get("lyric_pref_filter"):
    filtered = [(i, s) for (i, s) in hits if track_meta_by_index[i].get("lyr_emb") is not None]
    hits = filtered or hits

# 3) score candidates with theta = [wbpm, wlyrics, waudio]
scored: List[Tuple[int, float, Dict]] = []
for i, ann_score in hits:
    track_data = track_meta_by_index[i]
    s0, parts = score_track(track_data, query, theta)
    scored.append((i, s0, parts))

scored.sort(key=lambda x: x[1], reverse=True)
top = scored[:10]

for i, s, p in top:
    print(i, "score:", round(s, 4), "w_used:", tuple(round(x, 2) for x in p["w_used"]),
          "Sbpm:", round(p["Sbpm"], 3), "Slyrics:", round(p["Slyrics"], 3), "Saudio:", round(p["Saudio"], 3))
