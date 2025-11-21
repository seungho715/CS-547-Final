from feature_store import FeatureStore
from candidate_gen import generate_candidates
from scorer import score_track
from bandit_adapter import SoftmaxBandit
from typing import Dict, Any, List, Tuple

# --- Initialization ---
fs = FeatureStore("dataset/artifacts")

# Example query profile
query: Dict[str, Any] = {"bpm": 128.0, "delta": 6, "lyr_emb": None, "aud_emb": None}

# Initial bandit parameters
slider_w, alpha = 0.7, 0.1
bandit = SoftmaxBandit([slider_w, 1-slider_w, alpha], epsilon=0.2)

# --- Define Track Fetcher ---

def fetch_track(i: int) -> Dict[str, Any]:
    """Retrieves all track data (metadata and embeddings) needed for scoring."""
    # i is the integer index returned by the FAISS search
    track_id = fs.ids[i] 
    
    # Use the FeatureStore's new method to get the complete track data dictionary
    return fs.get_track_data(track_id) 

# --- Recommendation Logic ---

# 1. Pick the dynamic weighting strategy (arm)
arm_idx, theta = bandit.pick_arm()

# 2. Generate and pre-filter candidate tracks
# Pass the FeatureStore instance 'fs' to the generator
hits = generate_candidates(fs, query, k_ann=300, delta=6)

# 3. Score all candidates using the chosen weights (theta)
scored: List[Tuple[int, float, Dict]] = []
for i, ann_score in hits:
    # Fetch the full track data (including tempo, lyr_emb, aud_emb)
    track_data = fetch_track(i)
    
    # Score the track using the dynamic weights from the bandit
    s0, parts = score_track(track_data, query, w=theta[0], alpha=theta[2])
    scored.append((i, s0, parts))

# 4. Select and display the top results
scored.sort(key=lambda x: x[1], reverse=True)
top = scored[:10]

# Display final results (FAISS index, final score, breakdown)
print([(i, s, p) for (i, s, p) in top])
