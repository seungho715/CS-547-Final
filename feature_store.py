from pathlib import Path
import json, faiss, numpy as np
from joblib import load

class FeatureStore:
    def __init__(self, art_dir="dataset/artifacts"):
        p = Path(art_dir)        

        self.index = faiss.read_index(str(p/"tracks.faiss"))
        self.ids = np.load(p/"ids.npy", allow_pickle=True) # Maps FAISS index to original track IDs
        self.scaler = load(p/"scaler.joblib")
        self.spec = json.load(open(p/"feature_spec.json"))
        self.cols = self.spec["numeric_cols"]
        self.col_w = self.spec["column_weights"]

        # NEW Components (for Dynamic Scoring) ---
        
        with open(p/"track_metadata.json", "r") as f:
            self.track_meta_map = json.load(f)

        # Load the lyrics embeddings 
        self.lyrics_embeddings = load(p/"lyrics_embeddings.joblib")
        
        # Load pre-computed audio embeddings:
        '''Maybe recommend from MVP? Leave audio embeddings absent and set alpha = 0. Scorer renomralizes weights per track '''
        try:
             self.audio_embeddings = load(p/"audio_embeddings.joblib")
        except:
             print("Warning: Could not load audio_embeddings. Setting to empty map.")
             self.audio_embeddings = {}
        

    def vectorize(self, feat_dict: dict) -> np.ndarray:
        
        q = np.array([feat_dict.get(c, 0.0) for c in self.cols], dtype="float32")
        w = np.array([self.col_w.get(c, 1.0) for c in self.cols], dtype="float32")
        q = (q * w).reshape(1, -1)
        q = self.scaler.transform(q).astype("float32")
        q /= (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
        return q

    def ann_search(self, qv: np.ndarray, k: int = 200):
        
        scores, idx = self.index.search(qv, k)
        return [(int(i), float(s)) for i, s in zip(idx[0], scores[0])]

    def ids_to_track_ids(self, int_indices):
        
        return [self.ids[i] for i in int_indices]

    # Track Lookup 
    def get_track_data(self, track_id: str) -> dict:
        """Retrieves track metadata and all embeddings by track ID."""
        data = self.track_meta_map.get(track_id, {})
        
        # Attach the lyrics embedding vector
        data["lyr_emb"] = self.lyrics_embeddings.get(track_id)
        
        # Attach the audio embedding vector
        data["aud_emb"] = self.audio_embeddings.get(track_id)
        
        return data
