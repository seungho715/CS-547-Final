import math
import numpy as np
from candidate_gen import bpm_distance

def sbpm(bq, bi, delta=6):
    d = bpm_distance(bq, bi)
    return max(0.0, 1.0 - d/delta)

def cosine(a, b):
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na*nb))

def score_track(track, query, w, alpha=0.1):
    s_bpm = sbpm(query["bpm"], track.get("tempo", 0.0), delta=query.get("delta", 6))
    s_lyrics = cosine(query.get("lyr_emb"), track.get("lyr_emb")) if track.get("lyr_emb") is not None else 0.0
    s_audio  = cosine(query.get("aud_emb"), track.get("aud_emb")) if track.get("aud_emb") is not None else 0.0
    s0 = w*s_bpm + (1-w)*s_lyrics + alpha*s_audio
    return s0, {"Sbpm": s_bpm, "Slyrics": s_lyrics, "Saudio": s_audio}
