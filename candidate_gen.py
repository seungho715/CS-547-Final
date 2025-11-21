import numpy as np

def bpm_distance(bq, bi):
    return min(abs(bq-bi), abs(2*bq-bi), abs(bq-2*bi))

def bpm_prefilter(track_meta, bpm_target, delta=6):
    allowed = []
    for i, m in track_meta.items():
        if "tempo" in m and bpm_distance(bpm_target, m["tempo"]) <= delta:
            allowed.append(i)
    return set(allowed)

def mmr_rerank(candidates, track_meta, lambda_param=0.7):
    """MMR re-ranking: balance relevance and diversity"""
    if len(candidates) <= 1:
        return candidates

    selected = []
    remaining = candidates.copy()

    while remaining and len(selected) < len(candidates):
        best_score = -float('inf')
        best_idx = 0

        for idx, (track_id, relevance) in enumerate(remaining):
            # Compute max similarity with already selected tracks (redundancy)
            max_sim = 0.0
            if selected and track_id in track_meta:
                cand_tempo = track_meta[track_id].get("tempo", 0)
                for sel_id, _ in selected:
                    if sel_id in track_meta:
                        sel_tempo = track_meta[sel_id].get("tempo", 0)
                        # Use BPM distance to compute similarity
                        sim = max(0.0, 1.0 - bpm_distance(cand_tempo, sel_tempo) / 30.0)
                        max_sim = max(max_sim, sim)

            # MMR = λ × relevance - (1-λ) × redundancy
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        selected.append(remaining.pop(best_idx))

    return selected

def generate_candidates(fs, query_profile, track_meta, k_ann=300, delta=6, use_mmr=False, lambda_mmr=0.7):
    """
    Generate candidate tracks
    - use_mmr: whether to use MMR re-ranking for diversity
    - lambda_mmr: MMR λ parameter (0.0=diversity only, 1.0=relevance only)
    """
    qv = fs.vectorize({"tempo": query_profile["bpm"],
                       "energy": query_profile.get("energy", 0.8),
                       "valence": query_profile.get("valence", 0.5)})

    # Oversample candidates to ensure enough remain after BPM filtering
    oversample = 3 if not use_mmr else 5
    base_hits = fs.ann_search(qv, k_ann * oversample)

    allowed = bpm_prefilter(track_meta, query_profile["bpm"], delta)
    hits = [(i, s) for (i, s) in base_hits if i in allowed]

    # If not enough after filtering, use all candidates
    if len(hits) < k_ann:
        hits = base_hits

    hits = hits[:k_ann]

    # Optional: MMR re-ranking
    if use_mmr and track_meta:
        hits = mmr_rerank(hits, track_meta, lambda_mmr)

    return hits 