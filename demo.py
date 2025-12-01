from __future__ import annotations
import argparse, random
from typing import List, Dict, Any, Tuple
import numpy as np

from feature_store import FeatureStore
from candidate_gen import generate_candidates
from scorer import score_track
from bandit_adapter import SoftmaxUCBWeightBandit

RANDOM_SEED = 42
MMR_LAMBDA = 0.7

def mean_unit(vecs: List[np.ndarray] | List[None]) -> np.ndarray | None:
    vecs = [v for v in vecs if v is not None]
    if not vecs:
        return None
    v = np.stack(vecs).mean(axis=0)
    return v / (np.linalg.norm(v) + 1e-12)

def rank_once(fs: FeatureStore,
              track_meta_by_index: Dict[int, Dict[str, Any]],
              query: Dict[str, Any],
              theta: np.ndarray,
              k_ann: int,
              delta_bpm: int,
              use_mmr: bool = True,
              lambda_mmr: float = MMR_LAMBDA) -> List[Tuple[int, float, Dict[str, Any]]]:
    hits = generate_candidates(
        fs,
        query_profile=query,
        track_meta=track_meta_by_index,
        k_ann=k_ann,
        delta=delta_bpm,
        use_mmr=use_mmr,
        lambda_mmr=lambda_mmr,
    )
    
    if theta[1] > theta[0]:
        filtered = [(i, s) for (i, s) in hits if track_meta_by_index[i].get("lyr_emb") is not None]
        hits = filtered or hits

    scored = []
    for i, _ in hits:
        s, parts = score_track(track_meta_by_index[i], query, theta)
        scored.append((i, s, parts))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored

def main():
    parser = argparse.ArgumentParser(description="CLI demo for song recommendation")
    parser.add_argument("--art_dir", default="dataset/artifacts", help="artifacts directory")
    parser.add_argument("--seeds", default="", help="comma-separated track_ids to seed the query")
    parser.add_argument("--bpm", type=float, default=128.0, help="target BPM")
    parser.add_argument("--delta", type=int, default=6, help="BPM window")
    parser.add_argument("--w_bpm", type=float, default=0.7, help="slider BPM weight (0..1)")
    parser.add_argument("--alpha", type=float, default=0.0, help="audio weight (keep 0.0 unless you have audio embeddings)")
    parser.add_argument("--topk", type=int, default=10, help="how many results to print")
    parser.add_argument("--k_ann", type=int, default=300, help="ANN candidates before re-ranking")
    args = parser.parse_args()

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    fs = FeatureStore(args.art_dir)
    track_meta_by_index = {i: fs.get_track_data(fs.ids[i]) for i in range(len(fs.ids))}

    query: Dict[str, Any] = {
        "bpm": float(args.bpm),
        "delta": int(args.delta),
        "lyr_emb": None,
        "aud_emb": None,
    }

    # Build query lyrics embedding
    if args.seeds:
        seed_ids = [s.strip() for s in args.seeds.split(",") if s.strip()]
        seed_vecs = []
        for tid in seed_ids:
            where = np.where(fs.ids == tid)[0]
            if len(where):
                trk = track_meta_by_index[int(where[0])]
                seed_vecs.append(trk.get("lyr_emb"))
        query["lyr_emb"] = mean_unit(seed_vecs)
    else:
        eligible = [i for i in range(len(fs.ids)) if track_meta_by_index[i].get("lyr_emb") is not None]
        if len(eligible) >= 3:
            seeds = random.sample(eligible, 3)
            seed_vecs = [track_meta_by_index[i]["lyr_emb"] for i in seeds]
            query["lyr_emb"] = mean_unit(seed_vecs)
    # Initialize bandit with slider weights
    w_bpm = float(args.w_bpm)
    alpha = float(args.alpha)
    base = np.array([w_bpm, max(0.0, 1.0 - w_bpm - alpha), alpha], dtype=np.float32)
    bandit = SoftmaxUCBWeightBandit(base_weights=base.tolist(), eps=0.2, rng_seed=RANDOM_SEED)
    arm_idx, theta = bandit.pick_arm()
    ranked = rank_once(fs, track_meta_by_index, query, theta, k_ann=args.k_ann, delta_bpm=args.delta)
    print(f"\nSlider base=[{base[0]:.2f} BPM, {base[1]:.2f} Lyrics, {base[2]:.2f} Audio]  →  theta={tuple(round(x,2) for x in theta)}")
    print(f"Query: BPM={query['bpm']}, delta={query['delta']}, lyrics_emb={'yes' if query['lyr_emb'] is not None else 'no'}")
    print("-"*98)
    print(f"{'rank':<4}  {'track_id':<24}  {'score':>7}  {'Sbpm':>6}  {'Slyrics':>8}  {'w_used':>16}")
    print("-"*98)
    for rank, (i, s, parts) in enumerate(ranked[:args.topk], start=1):
        tid = fs.ids[i]
        w_used = tuple(round(x, 2) for x in parts["w_used"])
        print(f"{rank:<4}  {tid:<24}  {s:7.4f}  {parts['Sbpm']:6.3f}  {parts['Slyrics']:8.3f}  {str(w_used):>16}")

if __name__ == "__main__":
    main()
