from feature_store import FeatureStore
from candidate_gen import generate_candidates
from scorer import score_track
from bandit_adapter import SoftmaxBandit

track_meta = {} 

fs = FeatureStore("dataset/artifacts")
query = {"bpm": 128.0, "delta": 6, "lyr_emb": None, "aud_emb": None}
slider_w, alpha = 0.7, 0.1
bandit = SoftmaxBandit([slider_w, 1-slider_w, alpha], epsilon=0.2)

arm_idx, theta = bandit.pick_arm()
hits = generate_candidates(fs, query, track_meta, k_ann=300, delta=6)

def fetch_track(i):
    return {"tempo": track_meta.get(i, {}).get("tempo", 0.0),
            "lyr_emb": None, "aud_emb": None}

scored = []
for i, ann_score in hits:
    s0, parts = score_track(fetch_track(i), query, w=theta[0], alpha=theta[2])
    scored.append((i, s0, parts))

scored.sort(key=lambda x: x[1], reverse=True)
top = scored[:10]
print([(i, s, p) for (i, s, p) in top])
