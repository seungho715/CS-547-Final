# scorer.py
import numpy as np
from candidate_gen import bpm_distance

def sbpm(bq, bi, delta=6):
    d = bpm_distance(bq, bi)
    return max(0.0, 1.0 - d / delta)

def cosine(a, b):
    if a is None or b is None:
        return 0.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def score_track(track, query, theta):
    wbpm, wlyr, waud = map(float, theta)
    has_lyr = (query.get("lyr_emb") is not None) and (track.get("lyr_emb") is not None)
    has_aud = (query.get("aud_emb") is not None) and (track.get("aud_emb") is not None)
    if not has_lyr: wlyr = 0.0
    if not has_aud: waud = 0.0
    total_w = wbpm + wlyr + waud
    if total_w > 0:
        wbpm, wlyr, waud = wbpm/total_w, wlyr/total_w, waud/total_w

    Sbpm = sbpm(query["bpm"], track.get("tempo", 0.0), delta=query.get("delta", 6))
    Slyr = cosine(query.get("lyr_emb"), track.get("lyr_emb")) if has_lyr else 0.0
    Saud = cosine(query.get("aud_emb"), track.get("aud_emb")) if has_aud else 0.0

    S = wbpm*Sbpm + wlyr*Slyr + waud*Saud
    return S, {"Sbpm": Sbpm, "Slyrics": Slyr, "Saudio": Saud, "w_used": (wbpm, wlyr, waud)}
