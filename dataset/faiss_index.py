import json
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import dump
from sklearn.preprocessing import StandardScaler
import faiss
from charset_normalizer import from_path

CSV_PATH = "merged_dataset.csv"
OUT_DIR = Path("artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def read_csv_multilingual(path: str) -> pd.DataFrame:
    candidates = ["utf-8-sig", "utf-8", "cp932", "cp949", "gb18030", "cp1252", "latin-1"]
    errors = []
    for enc in candidates:
        try:
            return pd.read_csv(path, encoding=enc, engine="python")
        except UnicodeDecodeError as e:
            errors.append((enc, str(e)))
    return pd.read_csv(path, encoding="utf-8", encoding_errors="replace", engine="python")

df = read_csv_multilingual(CSV_PATH)

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

if "track_id" not in df.columns:
    raise ValueError("Missing 'track_id' column.")
df = df.drop_duplicates(subset=["track_id"]).reset_index(drop=True)

if "explicit" in df.columns:
    df["explicit"] = (
        df["explicit"]
        .map({True: 1, False: 0, "true": 1, "false": 0, 1: 1, 0: 0})
        .fillna(0)
        .astype(int)
    )

NUMERIC_COLS = [
    "danceability","energy","key","loudness","mode","speechiness",
    "acousticness","instrumentalness","liveness","valence","tempo",
    "time_signature","duration_ms","popularity","explicit"
]
NUMERIC_COLS = [c for c in NUMERIC_COLS if c in df.columns]

work = df.copy()

for c in NUMERIC_COLS:
    work[c] = pd.to_numeric(work[c], errors="coerce")
    if work[c].isna().any():
        work[c] = work[c].fillna(work[c].median())

X = work[NUMERIC_COLS].to_numpy(dtype="float32")
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
X *= w

scaler = StandardScaler(with_mean=True, with_std=True)
Xz = scaler.fit_transform(X).astype("float32")
norms = np.linalg.norm(Xz, axis=1, keepdims=True) + 1e-12
Xn = (Xz / norms).astype("float32")

d = Xn.shape[1]
index = faiss.IndexFlatIP(d)
index.add(Xn)

faiss.write_index(index, str(OUT_DIR / "tracks.faiss"))
np.save(OUT_DIR / "ids.npy", track_ids)
dump(scaler, OUT_DIR / "scaler.joblib")
with open(OUT_DIR / "feature_spec.json", "w") as f:
    json.dump(
        {
            "csv_path": CSV_PATH,
            "numeric_cols": NUMERIC_COLS,
            "column_weights": COLUMN_WEIGHTS,
            "similarity": "cosine via inner-product on L2-normalized vectors",
        },
        f,
        indent=2,
    )

meta_cols = ["track_id", "tempo", "energy", "valence", "duration_ms", "track_genre"]
meta_cols = [c for c in meta_cols if c in df.columns]
track_meta = df[meta_cols].to_dict(orient="records")
meta_map = {row["track_id"]: {k: row.get(k) for k in row if k != "track_id"} for row in track_meta}
with open(OUT_DIR / "track_metadata.json", "w") as f:
    json.dump(meta_map, f)

print(f"Built FAISS index with {len(track_ids)} tracks, dim={d}. Artifacts in {OUT_DIR}/")
