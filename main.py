from __future__ import annotations
from typing import Dict, Any, List, Tuple
import random
import numpy as np

from feature_store import FeatureStore
from candidate_gen import generate_candidates
from scorer import score_track
from bandit_adapter import SoftmaxUCBWeightBandit
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
@app.route('/')
def home():
    return "Hello, this is the music recommendation system!"

@app.route('/process_data', methods=['POST'])
def process_data():
        # 1. Accept Request Data
    if request.is_json:
        data = request.get_json()
    else:
        data = request.form # For form data

    # Example: Extracting a value from the received data
    input_value = data.get('some_key', 'default_value')

    # 2. Run Code
    # This is where your custom logic goes.
    # For demonstration, let's just manipulate the input_value.
    processed_value = f"Processed: {input_value.upper()}"

    # 3. Return Data
    response_data = {
        "status": "success",
        "original_input": input_value,
        "processed_output": processed_value
    }
    return jsonify(response_data)

@app.route('/recommend', methods=['POST'])
def create_model_request():
    """
    Initial request to set the base mood (bandit weight) and seed the query
    with an initial song, then returns the first list of recommendations.
    """
    global bandit, accepted_indices, query, RANDOM_SEED
    
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
        
    data = request.get_json()
    mood_value = data.get('mood')
    song_id = data.get('songId')

    if mood_value is None or song_id is None:
        return jsonify({"error": "Missing 'mood' (number) or 'songId' (string)"}), 400

    # 1. Reset Session State
    accepted_indices.clear()
    global per_track_penalty
    per_track_penalty.clear()
    
    # 2. Adjust Bandit Base Weight based on Mood (equivalent to on_slider_change)
    # Assuming mood_value is between 0.0 and 1.0, where 1.0 is highest weight for the first feature (e.g., lyrical content/vibe)
    new_w = float(mood_value)
    alpha = 0.0 # Assuming alpha remains 0.0
    new_base = [new_w, max(0.0, 1.0 - new_w - alpha), alpha]
    #bandit.set_base(new_base) # Use set_base if available, otherwise re-create #TODO: when creating model, getting error: AttributeError: 'SoftmaxUCBWeightBandit' object has no attribute 'set_base'
    # Re-create bandit if set_base is not implemented in SoftmaxUCBWeightBandit
    bandit = SoftmaxUCBWeightBandit(new_base, eps=0.2, rng_seed=RANDOM_SEED)

    # 3. Seed Query from initial song
    seed_from_frontend([song_id]) # seed_from_frontend is an existing function

    # 4. Get Recommendations
    arm_idx, theta = bandit.pick_arm()
    recommendations = get_recommendations(theta)

    return jsonify({"recommendations": recommendations})


@app.route('/adjust_mood', methods=['POST'])
def adjust_mood_request():
    """
    Adjusts the global bandit model's base weight (mood) and returns
    a new list of songs based on the new weights.
    """
    global bandit
    
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
        
    data = request.get_json()
    mood_value = data.get('mood')

    if mood_value is None:
        return jsonify({"error": "Missing 'mood' (number)"}), 400

    # 1. Adjust Bandit Base Weight based on Mood
    new_w = float(mood_value)
    alpha = 0.0
    new_base = [new_w, max(0.0, 1.0 - new_w - alpha), alpha]
    bandit.set_base(new_base) # Or re-create bandit as above #TODO: this function is giving set_base os not an attribute error (no set_base function in bandit_adapter)

    # 2. Get Recommendations
    arm_idx, theta = bandit.pick_arm()
    recommendations = get_recommendations(theta)

    return jsonify({"recommendations": recommendations})

@app.route('/feedback', methods=['POST'])
def like_or_skip_request():
    """
    Adjusts the global model with like or skip feedback, updates the bandit,
    and returns the next single recommendation.
    """
    global per_track_penalty, accepted_indices, pending_rewards, bandit
    
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
        
    data = request.get_json()
    feedback = data.get('likeOrSkip', '').lower()
    
    # Required for bandit update and history
    # NOTE: The track that was just played/skipped should be passed by the UI.
    track_index = data.get('trackIndex') 
    completion_ratio = data.get('completionRatio', 0.0) # Assume 0.0 if not provided
    skip_latency_s = data.get('skipLatencyS', 999.0) # Assume high value if not provided

    if feedback not in ['like', 'skip'] or track_index is None:
        return jsonify({"error": "Invalid 'likeOrSkip' or missing 'trackIndex'"}), 400

    # 1. Calculate Reward
    r = reward_from_event(completion_ratio, skip_latency_s)
    
    # 2. Update Session State (accepted/penalty)
    if feedback == 'like':
        accepted_indices.append(track_index)
        # Assuming the arm that generated the *last* recommendation is the one to reward
        # This is a simplification; a more robust system would track the arm used per song.
        # We'll use the arm that generated the *current* pick in the next step.
        pending_rewards.append((0, r)) # Placeholder arm_idx 0; will fix in the loop below
    elif feedback == 'skip':
        # Add penalty for skipped track
        per_track_penalty[track_index] = per_track_penalty.get(track_index, 0.0) + SESSION_PENALTY
        # For a skip, we don't update accepted_indices or the main bandit reward queue

    # 3. Bandit Update (if enough rewards are pending)
    # The actual arm index used must be recorded when the track was served.
    # Since we can't track it easily here, we will *only* update the bandit if we 
    # use the simplification: only update on 'like' and assume the last used arm was '0'.
    if len(pending_rewards) >= BANDIT_UPDATE_EVERY and feedback == 'like':
        # NOTE: This uses arm index '0' for simplicity. A better system stores the 
        # actual arm_idx when the track was served.
        mean_r = float(np.mean([x[1] for x in pending_rewards]))
        bandit.update(0, mean_r) # Update with mean reward for arm 0
        pending_rewards.clear()
        refresh_query_from_last_N()

    # 4. Get the Next Recommendation
    arm_idx, theta = bandit.pick_arm()
    recommendations = get_recommendations(theta, top_n=1) # Get only the top single track

    if not recommendations:
        return jsonify({"recommendations": [], "message": "No more unique candidates found."})

    # NOTE: If you need to record the arm used, you should store the arm_idx here
    # with the recommended track's index for use in the *next* feedback request.
    
    return jsonify({"recommendations": recommendations})

@app.route('/getSongs', methods=['GET'])
def get_songs():
    """
    Returns a list of selectable songs
    """
    global fs, track_meta_by_index

    song_list = []
    # iterate through all tracks in the feature store
    for i in range(len(fs.ids)):
        track_id = fs.ids[i]
        track_data = track_meta_by_index.get(i)
        
        #fixed issue here where all songs respond with correct track ids but names as 'Unknown Track' and artist as 'Unknown Artist'
        if track_data:
            song_list.append({
                "track_id": track_id,
                "track_name": track_data.get("name", "Unknown Track"),
                "artist_name": track_data.get("artist", "Unknown Artist")})
    
    return jsonify(song_list)

if __name__ == '__main__':
    app.run(debug=True) # debug=True enables auto-reloading and debugger



'''@app.route('/seed', methods=['POST'])
def seed():
    data = request.json
    track_ids = data.get('track_ids', [])
    seed_from_frontend(track_ids)
    return jsonify({"status": "ok", "message": "Seed tracks accepted"})

@app.route('/next', methods=['GET'])
def next_track():
    arm_idx, theta = bandit.pick_arm()
    ranked = rank_once(theta)
    top_i, top_score, parts = ranked[0]
    response = {
        "track_index": fs.ids[top_i],
        "score": float(top_score),
        "theta": [float(x) for x in theta],
        "components": parts
        #"w_used": parts['w_used'],
        #"Sbpm": parts['Sbpm'],
        #"Slyrics": parts['Slyrics'],
        #"Saudio": parts['Saudio']
    }
    return jsonify(response)

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.json
    track_id = data["track_id"]
    completion_ratio = data['completion_ratio']
    skip_latency_s = data['skip_latency_s']

    # Find the internal index
    idx = int(np.where(fs.ids == track_id)[0][0])
    
    r = reward_from_event(completion_ratio, skip_latency_s)
    history.append({
        "track_index": idx,
        "completion_ratio": completion_ratio,
        "skip_latency_s": skip_latency_s,
        "reward": r,
    })

    if completion_ratio < 0.5:  # assuming skip if less than 50% completed
        per_track_penalty[idx] = per_track_penalty.get(idx, 0.0) + SESSION_PENALTY
    
    else:
        accepted_indices.append(idx)
        pending_rewards.append((0, r))
        message = "Track accepted"

    if len(pending_rewards) >= BANDIT_UPDATE_EVERY:
        mean_r = float(np.mean([x[1] for x in pending_rewards]))
        bandit.update(pending_rewards[-1][0], mean_r)
        pending_rewards.clear()
        refresh_query_from_last_N()

    return jsonify({"status": "ok", "message": message})

'''

# Minimal config
RANDOM_SEED = 42
TOPK_CANDIDATES = 300
MMR_LAMBDA = 0.7
DELTA_BPM_DEFAULT = 6
EPS_GREEDY_ON_SKIP = 0.15
SESSION_PENALTY = 0.15
BANDIT_UPDATE_EVERY = 3
HISTORY_N_FOR_QUERY = 3

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

def reward_from_event(completion_ratio: float,
                      skip_latency_s: float,
                      k_early: float = 5.0,
                      lam: float = 0.5) -> float:
    # FRONTEND  provide inputs
    r = completion_ratio - lam * int(skip_latency_s < k_early)
    return max(-1.0, min(1.0, r))

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
    global query
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

# Helper for ranking using global components
def get_recommendations(theta: np.ndarray, top_n: int = 10) -> List[Dict[str, Any]]:
    ranked = rank_once(theta)
    results: List[Dict[str, Any]] = []

    unique_candidates = []
    seen_ids = set(accepted_indices)

    for i, score, parts in ranked:
        if i not in seen_ids:
            unique_candidates.append((i, score, parts))
        if len(unique_candidates) >= top_n:
            break
            
    for i, score, parts in unique_candidates:
        track_data = track_meta_by_index[i]
        results.append({
            "track_id": fs.ids[i],
            "track_name": track_data.get("name", "Unknown Track"),
            "artist_name": track_data.get("artist", "Unknown Artist"),
            "score": float(score) # Convert numpy float to native float for jsonify
        })
    return results


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
