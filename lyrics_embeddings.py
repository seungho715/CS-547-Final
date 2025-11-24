import os, json, math
from pathlib import Path
import numpy as np
import pandas as pd

CSV_PATH = "dataset/merged_dataset.csv"
OUT_DIR  = Path("dataset/artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "lyrics_embeddings.joblib"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 64
MAX_CHARS  = 4000
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64 

def read_csv_multilingual(path: str) -> pd.DataFrame:
    candidates = ["utf-8-sig", "utf-8", "cp932", "cp949", "gb18030", "cp1252", "latin-1"]
    for enc in candidates:
        try:
            return pd.read_csv(path, encoding=enc, engine="python")
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="utf-8", encoding_errors="replace", engine="python")

df = read_csv_multilingual(CSV_PATH).drop_duplicates(subset=["track_id"]).reset_index(drop=True)

if "lyrics" not in df.columns:
    raise ValueError("CSV lacks 'lyrics' column needed for lyric embeddings.")

from sentence_transformers import SentenceTransformer
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL_NAME, device=device)

def chunk_text(s: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    s = s or ""
    if len(s) <= size:
        return [s]
    out = []
    i = 0
    while i < len(s) and len(out) < math.ceil(MAX_CHARS / size) + 2:
        out.append(s[i:i+size])
        i += (size - overlap)
    return out

emb_map = {}

texts = []
ids = []
for _, row in df.iterrows():
    tid = str(row["track_id"])
    lyr = str(row["lyrics"]) if not pd.isna(row["lyrics"]) else ""
    if not lyr.strip():
        continue
    chunks = chunk_text(lyr[:MAX_CHARS])
    texts.append(chunks)
    ids.append(tid)

flat_texts = [c for chunks in texts for c in chunks]
embs = model.encode(flat_texts, batch_size=BATCH_SIZE, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

ptr = 0
for tid, chunks in zip(ids, texts):
    n = len(chunks)
    vec = embs[ptr:ptr+n].mean(axis=0) if n > 0 else None
    ptr += n
    if vec is not None and np.isfinite(vec).all():
        emb_map[tid] = vec.astype("float32")

from joblib import dump
dump(emb_map, OUT_PATH)
print(f"Saved {len(emb_map)} lyric embeddings → {OUT_PATH}")
