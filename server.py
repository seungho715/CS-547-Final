from __future__ import annotations
import argparse, random
from typing import List, Dict, Any, Tuple
import numpy as np
from feature_store import FeatureStore
from candidate_gen import generate_candidates
from scorer import score_track
from bandit_adapter import SoftmaxUCBWeightBandit
from reward_policy import calculate_reward
from flask import Flask, request, jsonify
from flask_cors import CORS
import random
from fuzzywuzzy import fuzz


RANDOM_SEED = 42
TOPK_CANDIDATES = 300
MMR_LAMBDA = 0.7
DELTA_BPM_DEAFAULT = 6
EPS_GREEDY_ON_SKIP = 0.15
SESSION_PENALTY = 0.15
BANDIT_UPDATE_EVERY = 3
HISTORY_N_FOR_QUERY = 3
duration = meta.get("duration_ms")
TRACK_DURATION_S = float(duration) / 1000 if dur_ms is not None else 180. 


app = Flask(__name__)
CORS(app)

# hold all mutable states for session
# STATE = {
#     'per_track_penalty': {},        # Dict[int, float]
#     'accepted_indices': [],         # List[int]
#     'pending_rewards': [],          # List[Tuple[int, float]]
#     'query': {},                    # Dict[str, Any]
#     'fs': None,                     # FeatureStore object
#     'track_meta_by_index': None,    # Metadata dict
#     'bandit': None,                 # SoftmaxUCBWeightBandit object
#     'initial_base': [],             # Initial base weights
# }

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

fs = FeatureStore("dataset/artifacts")
track_meta_by_index = {i: fs.get_track_data(fs.ids[i]) for i in range(len(fs.ids))}

query: Dict[str, Any] = {"bpm": 128.0, "delta": 6, "lyr_emb": None, "aud_emb": None}

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
    global fs, track_meta_by_index

    song_list = []
    # iterate through all tracks in the feature store
    for i in range(len(fs.ids)):
        track_id = fs.ids[i]
        track_data = track_meta_by_index.get(i)
        
        #fixed issue here where all songs respond with correct track ids but names as 'Unknown Track' and artist as 'Unknown Artist'
        track_name = track_data.get("name", "Unknown Track")
        track_artist = track_data.get("artist", "Unknown Artist")
        if track_data and ((fuzz.ratio(track_name.lower(), search.lower()) >= 65) or (fuzz.ratio(track_artist.lower(), search.lower()) >= 65)):
            song_list.append({
                "track_id": track_id,
                "track_name": track_name,
                "artist_name": track_artist,
                "length_seconds": track_data.get("duration_ms", "") / 1000})
    
    return jsonify(song_list)

@app.route('/recommend', methods=['POST'])
def create_model_request():
    global bandit, query, fs, track_meta_by_index, arm_idx
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    
    data = request.get_json()
    mood_value = data.get('mood')
    song_id = data.get('songId')

    seed_ids = [s.strip() for s in song_id.split(",") if s.strip()]
    seed_vecs = []
    for tid in seed_ids:
        loc = np.where(fs.ids == tid)[0]
        if len(loc):
            seed_vecs.append(track_meta_by_index[int(loc[0])].get("lyr_emb"))
    query["lyr_emb"] = mean_unit(seed_vecs)

    alpha = 0.0

    w_bpm, alpha = float(mood_value), float(alpha)
    base = np.array([w_bpm, max(0.0, 1.0 - w_bpm - alpha), alpha], dtype=np.float32)
    bandit = SoftmaxUCBWeightBandit(base_weights=base.tolist(), eps=0.2, rng_seed=RANDOM_SEED)
    arm_idx, theta = bandit.pick_arm()

    ranked = rank_once(fs, track_meta_by_index, query, theta, k_ann=300, delta_bpm=6)

    return jsonify(ranked)

@app.route('/adjust_mood', methods=['POST'])
def adjust_mood_request():
    global bandit, arm_idx
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json()
    mood_value = data.get('mood')

    if mood_value is None:
        return jsonify({"error": "Missing 'mood' (number)"}), 400

    alpha = 0.0

    w_bpm, alpha = float(mood_value), float(alpha)
    new_base = np.array([w_bpm, max(0.0, 1.0 - w_bpm - alpha), alpha], dtype=np.float32)
    bandit.set_base(new_base) #TODO: may need a setter for base

    # Get Recommendations
    arm_idx, theta = bandit.pick_arm()
    ranked = rank_once(fs, track_meta_by_index, query, theta, k_ann=300, delta_bpm=6)

    return jsonify({"recommendations": ranked})

@app.route("/likeOrSkip", methods=['POST']) #TODO: just this needs to be finished
def likeOrSkip():
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json()
    songLiked = data.get("liked")
    song = data.get("song")
    top_i, top_score, parts = song #get song back TODO: figure out what top_i, top_score, and parts are in terms of sending back from UI

    if songLiked == "full":
        completion_ratio = 0.95 if args.cr is None else float(args.cr)
        skip_latency_s   = 999.0 if args.skip_s is None else float(args.skip_s)
        is_skipped       = False
    else: 
        completion_ratio = 0.05 if args.cr is None else float(args.cr)
        skip_latency_s   = 3.0  if args.skip_s is None else float(args.skip_s)
        is_skipped       = True

    r = calculate_reward(TRACK_DURATION_S, completion_ratio * TRACK_DURATION_S,
                        is_skipped=is_skipped, skip_latency_s=skip_latency_s)

    bandit.update(arm_idx, r)
    arm_idx2, theta2 = bandit.pick_arm()
    ranked2 = rank_once(fs, track_meta_by_index, query, theta2, k_ann=300, delta_bpm=6)

    return jsonify({"recommendations": ranked2})

def main():
    app.run(debug=True)

if __name__ == '__main__': # debug=True enables auto-reloading and debugger
    main()