import numpy as np

def bpm_distance(bq, bi):
    return min(abs(bq-bi), abs(2*bq-bi), abs(bq-2*bi))

def bpm_prefilter(track_meta, bpm_target, delta=6):
    allowed = []
    for i, m in track_meta.items():
        if "tempo" in m and bpm_distance(bpm_target, m["tempo"]) <= delta:
            allowed.append(i)
    return set(allowed)

def generate_candidates(fs, query_profile, track_meta, k_ann=300, delta=6):
    qv = fs.vectorize({"tempo": query_profile["bpm"],
                       "energy": query_profile.get("energy", 0.8),
                       "valence": query_profile.get("valence", 0.5)})
    base_hits = fs.ann_search(qv, k_ann) 
    allowed = bpm_prefilter(track_meta, query_profile["bpm"], delta)
    hits = [(i, s) for (i, s) in base_hits if i in allowed]
    return hits 