# build_index.py
import json
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
import faiss

CSV_PATH = "dataset.csv"
OUT_DIR = Path("artifacts")
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(CSV_PATH)

if "track_id" not in df.columns:
    raise ValueError("Missing 'track_id' column.")

NUMERIC_COLS = [
    "danceability","energy","key","loudness","mode","speechiness",
    "acousticness","instrumentalness","liveness","valence","tempo",
    "time_signature","duration_ms","popularity","explicit"
]

NUMERIC_COLS = [c for c in NUMERIC_COLS if c in df.columns]


work = df.copy()
for c in NUMERIC_COLS:
    if work[c].isna().any():
        work[c] = work[c].fillna(work[c].median())

work[NUMERIC_COLS] = work[NUMERIC_COLS].apply(pd.to_numeric, errors="coerce")
for c in NUMERIC_COLS:
    if work[c].isna().any():
        work[c] = work[c].fillna(work[c].median())

X = work[NUMERIC_COLS].to_numpy().astype("float32")
track_ids = work["track_id"].astype(str).to_numpy()

COLUMN_WEIGHTS = {
    "tempo": 1.5,
    "energy": 1.2,
    # "valence": 1.1,
}
w = np.ones(len(NUMERIC_COLS), dtype="float32")
for i, c in enumerate(NUMERIC_COLS):
    if c in COLUMN_WEIGHTS:
        w[i] = float(COLUMN_WEIGHTS[c])
X = X * w

scaler = StandardScaler(with_mean=True, with_std=True)
Xz = scaler.fit_transform(X).astype("float32")

norms = np.linalg.norm(Xz, axis=1, keepdims=True) + 1e-12
Xn = (Xz / norms).astype("float32")

# Build FAISS index 
d = Xn.shape[1]
index = faiss.IndexFlatIP(d)  
index.add(Xn) 

# Save artifacts 
faiss.write_index(index, str(OUT_DIR / "tracks.faiss"))
np.save(OUT_DIR / "ids.npy", track_ids)
dump(scaler, OUT_DIR / "scaler.joblib")
with open(OUT_DIR / "feature_spec.json", "w") as f:
    json.dump(
        {
            "numeric_cols": NUMERIC_COLS,
            "column_weights": COLUMN_WEIGHTS,
            "similarity": "cosine via inner-product on L2-normalized vectors",
        },
        f,
        indent=2,
    )

print(f"Built FAISS index with {len(track_ids)} tracks, dim={d}. Artifacts in {OUT_DIR}/")
