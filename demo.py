from __future__ import annotations
import argparse, random
from typing import List, Dict, Any, Tuple
import numpy as np

from feature_store import FeatureStore
from candidate_gen import generate_candidates
from scorer import score_track
from bandit_adapter import SoftmaxUCBWeightBandit
from reward_policy import calculate_reward

RANDOM_SEED = 42
MMR_LAMBDA = 0.7
TRACK_DURATION_S = 180.0  # demo default

def mean_unit(vecs: List[np.ndarray] | List[None]) -> np.ndarray | None:
    vecs = [v for v in vecs if v is not None]
    if not vecs: return None
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
    p = argparse.ArgumentParser(description="CLI demo for song recommendation + reward/bandit")
    p.add_argument("--art_dir", default="dataset/artifacts")
    p.add_argument("--seeds", default="", help="comma-separated track_ids to seed lyrics")
    p.add_argument("--bpm", type=float, default=128.0)
    p.add_argument("--delta", type=int, default=6)
    p.add_argument("--w_bpm", type=float, default=0.7)
    p.add_argument("--alpha", type=float, default=0.0)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--k_ann", type=int, default=300)

    # NEW: simulate outcome and update bandit
    p.add_argument("--simulate", choices=["none","full","skip"], default="none",
                   help="simulate a listening outcome for top-1")
    p.add_argument("--cr", type=float, default=None, help="override completion ratio [0,1]")
    p.add_argument("--skip_s", type=float, default=None, help="override skip latency seconds")
    p.add_argument("--update_bandit", action="store_true", help="apply reward update then re-rank")
    args = p.parse_args()

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    fs = FeatureStore(args.art_dir)
    track_meta_by_index = {i: fs.get_track_data(fs.ids[i]) for i in range(len(fs.ids))}

    query: Dict[str, Any] = {"bpm": args.bpm, "delta": args.delta, "lyr_emb": None, "aud_emb": None}

    if args.seeds:
        seed_ids = [s.strip() for s in args.seeds.split(",") if s.strip()]
        seed_vecs = []
        for tid in seed_ids:
            loc = np.where(fs.ids == tid)[0]
            if len(loc):
                seed_vecs.append(track_meta_by_index[int(loc[0])].get("lyr_emb"))
        query["lyr_emb"] = mean_unit(seed_vecs)
    else:
        eligible = [i for i in range(len(fs.ids)) if track_meta_by_index[i].get("lyr_emb") is not None]
        if len(eligible) >= 3:
            seeds = random.sample(eligible, 3)
            query["lyr_emb"] = mean_unit([track_meta_by_index[i]["lyr_emb"] for i in seeds])

    w_bpm, alpha = float(args.w_bpm), float(args.alpha)
    base = np.array([w_bpm, max(0.0, 1.0 - w_bpm - alpha), alpha], dtype=np.float32)
    bandit = SoftmaxUCBWeightBandit(base_weights=base.tolist(), eps=0.2, rng_seed=RANDOM_SEED)
    arm_idx, theta = bandit.pick_arm()

    ranked = rank_once(fs, track_meta_by_index, query, theta, k_ann=args.k_ann, delta_bpm=args.delta)
    print(f"\nSlider base=[{base[0]:.2f} BPM, {base[1]:.2f} Lyrics, {base[2]:.2f} Audio]  →  theta={tuple(round(x,2) for x in theta)}")
    print(f"Query: BPM={query['bpm']}, delta={query['delta']}, lyrics_emb={'yes' if query['lyr_emb'] is not None else 'no'}")
    print("-"*102)
    print(f"{'rank':<4}  {'faiss_idx':<8}  {'track_id':<24}  {'score':>7}  {'Sbpm':>6}  {'Slyrics':>8}  {'Saudio':>7}  {'w_used':>16}")
    print("-"*102)
    for rank, (i, s, parts) in enumerate(ranked[:args.topk], start=1):
        tid = fs.ids[i]
        w_used = tuple(round(x, 2) for x in parts["w_used"])
        print(f"{rank:<4}  {i:<8}  {tid:<24}  {s:7.4f}  {parts['Sbpm']:6.3f}  {parts['Slyrics']:8.3f}  {parts['Saudio']:7.3f}  {str(w_used):>16}")

    if args.simulate != "none":
        top_i, top_score, parts = ranked[0]
        if args.simulate == "full":
            completion_ratio = 0.95 if args.cr is None else float(args.cr)
            skip_latency_s   = 999.0 if args.skip_s is None else float(args.skip_s)
            is_skipped       = False
        else: 
            completion_ratio = 0.05 if args.cr is None else float(args.cr)
            skip_latency_s   = 3.0  if args.skip_s is None else float(args.skip_s)
            is_skipped       = True

        r = calculate_reward(TRACK_DURATION_S, completion_ratio * TRACK_DURATION_S,
                             is_skipped=is_skipped, skip_latency_s=skip_latency_s)
        print(f"\n[simulate={args.simulate}] top_i={top_i}, reward={r:.4f}")

        if args.update_bandit:
            bandit.update(arm_idx, r)
            # Pick again to see drift
            arm_idx2, theta2 = bandit.pick_arm()
            ranked2 = rank_once(fs, track_meta_by_index, query, theta2, k_ann=args.k_ann, delta_bpm=args.delta)
            print(f"[after update] theta={tuple(round(x,2) for x in theta2)}; top idx={ranked2[0][0]} score={ranked2[0][1]:.4f}")

if __name__ == "__main__":
    main()
