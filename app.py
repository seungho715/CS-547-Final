from __future__ import annotations
import argparse, random
from typing import List, Dict, Any, Tuple
import numpy as np
import hashlib
from feature_store import FeatureStore
from candidate_gen import generate_candidates
from scorer import score_track
from bandit_adapter import SoftmaxUCBWeightBandit
from reward_policy import calculate_reward
from flask import Flask, request, jsonify
from flask_cors import CORS
import random
from fuzzywuzzy import fuzz
from collections import deque


RANDOM_SEED = 42
TOPK_CANDIDATES = 10
MMR_LAMBDA = 0.7
DELTA_BPM_DEFAULT = 6
EPS_GREEDY_ON_SKIP = 0.15
SESSION_PENALTY = 0.15
BANDIT_UPDATE_EVERY = 3
HISTORY_N_FOR_QUERY = 3
TRACK_DURATION_S = 180.0
RECENT_NO_REPEAT = 5
TOP_M = 3
TAU = 0.05


random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

fs = FeatureStore("dataset/artifacts")
track_meta_by_index = {i: fs.get_track_data(fs.ids[i]) for i in range(len(fs.ids))}

#query: Dict[str, Any] = {"bpm": 128.0, "delta": 6, "lyr_emb": None, "aud_emb": None}
per_track_penalty: Dict[int, float] = {}
accepted_indices: List[int] = []
pending_rewards: List[Tuple[int, float]] = []
bandit: SoftmaxUCBWeightBandit = None # Initialized in /recommend
query: Dict[str, Any] = {"bpm": 128.0, "delta": DELTA_BPM_DEFAULT, "lyr_emb": None, "aud_emb": None}
arm_idx_served: int = None # Stores the arm index used for the last served track
recent_served: deque[int] = deque(maxlen=RECENT_NO_REPEAT)

def mean_unit(vecs: List[np.ndarray] | List[None]) -> np.ndarray | None:
    vecs = [v for v in vecs if v is not None]
    if not vecs: 
        return None
    v = np.stack(vecs).mean(axis=0)
    n = np.linalg.norm(v)
    return v / (n + 1e-12)

def refresh_query_from_last_N():
    """Updates the global query lyric embedding based on the last N accepted tracks."""
    global accepted_indices, query, fs, track_meta_by_index
    
    last_N = accepted_indices[-HISTORY_N_FOR_QUERY:]
    if not last_N: return
    
    vecs = [track_meta_by_index[i].get("lyr_emb") for i in last_N]
    query["lyr_emb"] = mean_unit(vecs)

def pick_from_top_m(ranked: List[Tuple[int, float, Dict[str, Any]]], top_m: int = TOP_M, tau: float = TAU):
    if not ranked:
        raise RuntimeError("No candidates exist.")
    k = min(top_m, len(ranked))
    cand = ranked[:k]
    print(cand)
    scores = np.array([s["score"] for s in cand], dtype=np.float32)
    scores = scores - scores.max()
    probs = np.exp(scores / max(1e-6, tau))
    probs /= probs.sum()
    idx = int(np.random.choice(k, p=probs))
    return cand[idx]

def rank_once(fs, track_meta_by_index, query, theta, k_ann, delta_bpm) -> List[Dict[str, Any]]:
    """
    Performs the ranking, applies penalties, and formats the output for JSON.
    NOTE: This is the core logic from your main.py adapted for a utility function.
    """
    global per_track_penalty
    
    # 1. Candidate Generation
    hits = generate_candidates(
        fs,
        query_profile=query,
        track_meta=track_meta_by_index,
        k_ann=k_ann,
        delta=query.get("delta", delta_bpm),
        use_mmr=True,
        lambda_mmr=MMR_LAMBDA
    )

    if theta[1] > theta[0]:
        filtered = [(i, s) for (i, s) in hits if track_meta_by_index[i].get("lyr_emb") is not None]
        hits = filtered or hits

    hits = [(i, s) for (i, s) in hits if i not in recent_served]
    
    # 2. Score and Penalty Application
    scored: List[Tuple[int, float, Dict[str, Any]]] = []
    for i, _ in hits:
        s, parts = score_track(track_meta_by_index[i], query, theta)
        s -= per_track_penalty.get(i, 0.0) # Apply penalty
        scored.append((i, s, parts))
        
    scored.sort(key=lambda x: x[1], reverse=True)
    
    # 3. Format output for JSON
    recommendations = []
    for top_i, top_score, parts in scored:
        track_data = track_meta_by_index[top_i]
        recommendations.append({
            "track_index": int(top_i),
            "track_id": fs.ids[top_i],
            "score": float(top_score),
            "track_name": track_data.get("track_name"),
            "artist_name": track_data.get("artists"),
            "parts": {k: float(v) for k, v in parts.items() if k != 'w_used'},
            "theta_used": [round(float(x), 4) for x in theta]
        })
        
    return recommendations

app = Flask(__name__)
CORS(app, allow_headers=['Content-Type', 'ngrok-skip-browser-warning'])

@app.route('/')
def home():
    return "Hello, this is the music recommendation system!"

@app.route('/searchSongs', methods=['POST'])
def search_songs():
    """
    Returns a list of selectable songs
    """
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
        
    data = request.get_json()
    search = data.get('search', '').lower()
    #global fs, track_meta_by_index

    song_list = []
    # iterate through all tracks in the feature store
    for i in range(len(fs.ids)):
        track_id = fs.ids[i]
        track_data = track_meta_by_index.get(i)
        #fixed issue here where all songs respond with correct track ids but names as 'Unknown Track' and artist as 'Unknown Artist'
        track_name = str(track_data.get("track_name", "Unknown Track"))
        track_artist = str(track_data.get("artists", "Unknown Artist"))
        if track_data: #and (search.lower() in track_name.lower()) or (search.lower() in track_artist.lower()):
            name_score = fuzz.ratio(track_name.lower(), search)
            artist_score = fuzz.ratio(track_artist.lower(), search)
            if name_score > 65 or artist_score > 65:
                song_list.append({
                    "track_id": track_id,
                    "track_name": track_name,
                    "artist_name": track_artist,})
                    #"length_seconds": TRACK_DURATION_S,
    return jsonify(song_list)

@app.route('/recommend', methods=['POST'])
def create_model_request():
    global bandit, query, arm_idx_served, accepted_indices, per_track_penalty, pending_rewards, fs, track_meta_by_index, arm_idx
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    
    data = request.get_json()
    mood_value = data.get('mood')
    song_id = data.get('songId')

    if mood_value is None or song_id is None:
        return jsonify({"error": "Missing 'mood' (number) or 'songId' (string)"}), 400
    
    accepted_indices.clear()
    per_track_penalty.clear()   # Reset session state
    pending_rewards.clear() 

    seed_ids = [s.strip() for s in song_id.split(",") if s.strip()]
    seed_vecs = []
    for tid in seed_ids:
        loc = np.where(fs.ids == tid)[0]
        if len(loc):
            seed_vecs.append(track_meta_by_index[int(loc[0])].get("lyr_emb"))
    query["lyr_emb"] = mean_unit(seed_vecs)

    alpha = 0.0

    w_bpm, alpha = float(mood_value), float(alpha)

    if song_id:
        seed_hash = int(hashlib.sha256(song_id.encode('utf-8')).hexdigest(), 16) % (2**32) 
        bandit_seed = seed_hash
    else:
        bandit_seed = RANDOM_SEED # Fallback

    base = np.array([float(w_bpm), max(0.0, 1.0 - w_bpm - alpha), alpha], dtype=np.float32)

    bandit = SoftmaxUCBWeightBandit(base_weights=base.tolist(), eps=0.2, rng_seed=bandit_seed)
    arm_idx, theta = bandit.pick_arm()
    arm_idx_served = arm_idx

    ranked = rank_once(fs, track_meta_by_index, query, theta, k_ann=TOPK_CANDIDATES, delta_bpm=DELTA_BPM_DEFAULT)
    if ranked:
        ranked[0]['arm_idx'] = int(arm_idx)

    formatted_ranked = []
    for song in ranked[:10]:
        print(song)
        formatted_ranked.append({
                "track_id": song['track_id'],
                "track_name": song['track_name'],
                "artist_name": song['artist_name']})

    return jsonify(formatted_ranked)

@app.route('/adjust_mood', methods=['POST'])
def adjust_mood_request():
    global bandit, arm_idx_served
    if bandit is None:
        return jsonify({"error": "Model not initialized. Call /recommend first."}), 400
    
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json()
    mood_value = data.get('mood')

    if mood_value is None:
        return jsonify({"error": "Missing 'mood' (number)"}), 400

    alpha = 0.0

    w_bpm, alpha = float(mood_value), float(alpha)
    new_base = np.array([w_bpm, max(0.0, 1.0 - w_bpm - alpha), alpha], dtype=np.float32)
    bandit.set_base(new_base.tolist()) #TODO: may need a setter for base

    # Get Recommendations
    arm_idx, theta = bandit.pick_arm()
    arm_idx_served = arm_idx
    ranked = rank_once(fs, track_meta_by_index, query, theta, k_ann=TOPK_CANDIDATES, delta_bpm=DELTA_BPM_DEFAULT)

    formatted_ranked = []
    for song in ranked[:10]:
        print(song)
        formatted_ranked.append({
                "track_id": song['track_id'],
                "track_name": song['track_name'],
                "artist_name": song['artist_name']})

    return jsonify(formatted_ranked)

@app.route("/likeOrSkip", methods=['POST']) #TODO: just this needs to be finished
def likeOrSkip():
    global bandit, query, fs, track_meta_by_index, per_track_penalty, accepted_indices, pending_rewards, arm_idx_served
    
    if bandit is None:
        return jsonify({"error": "Model not initialized. Call /recommend first."}), 400
    
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json()
    songLiked = data.get("liked")
    track_id_str = data.get("song")

    arm_idx = arm_idx_served

    if track_id_str is None or arm_idx is None:
        return jsonify({"error": "Song data missing 'track_id' or 'arm_idx'"}), 400

    #if not song_data:
        #return jsonify({"error": "Missing 'song' data from previous recommendation"}), 400
    
        # Assuming the client passed a dictionary containing the necessary backend IDs
    #track_id_str = song_data.get('track_id')
        
        # Look up the internal integer index
    where = np.where(fs.ids == track_id_str)[0]
    if len(where) == 0:
        return jsonify({"error": f"Track ID '{track_id_str}' not found."}), 400
    track_index_int = int(where[0])

    
    #top_i, top_score, parts = song #get song back TODO: figure out what top_i, top_score, and parts are in terms of sending back from UI

    if songLiked == "full":
        completion_ratio = 0.90
        skip_latency_s   = 180
        is_skipped       = False
    else: 
        completion_ratio = 0.05
        skip_latency_s   = 3.0 
        is_skipped       = True

    r = calculate_reward(TRACK_DURATION_S, completion_ratio * TRACK_DURATION_S,
                        is_skipped=is_skipped, skip_latency_s=skip_latency_s)

    if is_skipped:
        # Apply penalty for skipped track
        per_track_penalty[track_index_int] = per_track_penalty.get(track_index_int, 0.0) + SESSION_PENALTY
        print(f"[SESSION] Track {track_index_int} skipped. Penalty applied.")
        
    else: 
        # Liked/Full listen: Add to accepted history and pending rewards batch
        accepted_indices.append(track_index_int)
        pending_rewards.append((arm_idx, r))
        print(f"[SESSION] Track {track_index_int} accepted. Reward {r:.4f} pending.")
    
    bandit_updated = False
    if len(pending_rewards) >= BANDIT_UPDATE_EVERY:
        # Update the bandit with the mean reward for the batch
        mean_r = float(np.mean([x[1] for x in pending_rewards]))
        
        # Use the arm_idx of the *last* track in the batch for the update
        last_arm_idx_in_batch = pending_rewards[-1][0]
        
        bandit.update(last_arm_idx_in_batch, mean_r) # Update the MAB model
        
        pending_rewards.clear()
        refresh_query_from_last_N() # Update contextual query
        bandit_updated = True

    # --- 5. Get the Next Recommendation ---
    
    arm_idx_next, theta_next = bandit.pick_arm()
    arm_idx_served = arm_idx_next # Store new arm for next feedback cycle
    
    # Get the new ranking
    ranked2 = rank_once(fs, track_meta_by_index, query, theta_next, k_ann=TOPK_CANDIDATES, delta_bpm=DELTA_BPM_DEFAULT)

    top_i = pick_from_top_m(ranked2)
    recent_served.append(top_i) 

    # Attach the new arm index to the top recommendation
    if ranked2:
        ranked2[0]['arm_idx'] = int(arm_idx_next)
    
    formatted_ranked = []
    for song in ranked2[:10]:
        print(song)
        formatted_ranked.append({
                "track_id": song['track_id'],
                "track_name": song['track_name'],
                "artist_name": song['artist_name']})

    return jsonify({
        "recommendations": formatted_ranked,
        "bandit_updated": bandit_updated,
        "reward_received": r
    })

def main():
    app.run(debug=True)

if __name__ == '__main__': # debug=True enables auto-reloading and debugger
    main()