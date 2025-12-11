# Adaptive Tempo-Lyric Recommender (ATLR)
The Adaptive Tempo-Lyric Recommender (ATLR) is a dynamic music recommendation system designed to provide personalized, real-time track suggestions by fusing musical structure (Tempo/BPM) with semantic meaning (Lyrical Content).

ATLR is built on a hybrid architecture that combines efficient Approximate Nearest Neighbor (ANN) Retrieval with a Multi-Armed Bandit (MAB) for adaptive, session-based re-ranking.
## Key Features & Differentiators
Unlike traditional systems that rely heavily on collaborative filtering or generalized audio features, ATLR focuses on transparent user control and immediate session adaptation.

* User-Tunable Control: Provides a mechanism for users to set the relative base weights between Tempo/BPM and Lyrical content, making the recommendation bias transparent.
* Fast Session Adaptation (Softmax UCB): Utilizes a SoftmaxUCBWeightBandit to dynamically adjust the scoring weights (θ) after every 3-5 plays based on implicit feedback (e.g., track completion and skip latency), ensuring the recommendations adapt to the user's most up-to-date mood.
* Musical BPM Anchoring: Employs a custom bpm_distance function to filter and score tracks, correctly accounting for half-time and double-time equivalence (e.g., treating 60 BPM and 120 BPM as similar), which is crucial for musical coherence.
* Diversity (MMR Re-ranking): Applies Maximal Marginal Relevance (MMR) to balance the retrieved candidates between high relevance (ANN score) and maximal diversity (based on BPM similarity), reducing repetitive recommendations.
* Adaptive Fusion Scoring: Features a robust score_track function that automatically renormalizes the active feature weights if a track lacks certain features (e.g., lyrics or audio embeddings), preventing scoring bias.
## Architecture and Pipeline
The system operates in a five-stage loop, driven by user interaction and implicit feedback.

* Bandit Arm Selection: The SoftmaxUCBWeightBandit selects a weight vector θ = [wbpm, wlyrics, waudio] for the session, constrained to remain near the user's explicit slider preference.

* Candidate Generation:
1.  A query is run against the FAISS Index (built on 15 weighted numeric features like tempo=1.5, energy=1.2) for high-recall retrieval.

2. Candidates are pre-filtered using the tempo-octave-aware bpm_prefilter.

3. The remaining candidates are re-ranked using MMR to maximize diversity.

* Dynamic Scoring: The score_track function computes the final rank based on the bandit's weights θ and three similarity components:

Score =wbpm . Sbpm + wlyrics . Slyrics  + waudio . Saudio

Weights are adaptively normalized if data is missing.
* Implicit Feedback & Reward: After a track is played, an implicit reward is calculated based on session metrics (e.g., play time, skip latency). The reward policy penalizes early skips (e.g., skipping before 30 seconds).

* Bandit Update: The reward is fed back to the SoftmaxUCBWeightBandit via bandit.update(), adjusting the future selection probability of that weight arm to reinforce successful recommendations.
## Data and Feature Engineering
* FAISS Index: The core retrieval index is built using a faiss.IndexFlatIP (Inner Product, for cosine similarity on L2-normalized vectors) over 15 scaled audio features.
* Weighted Features: Features like tempo (1.5) and energy (1.2) are explicitly weighted up during the initial vectorization for the ANN search to reflect their importance.
* Lyric Embeddings: Lyrical content is processed using a pre-trained multilingual Sentence Transformer (paraphrase-multilingual-MiniLM-L12-v2) to generate semantic embeddings for similarity calculation.
## Getting Started
### Prerequisites
* Python (3.8+)
* Standard ML/data science libraries (numpy, pandas, scikit-learn, faiss, joblib)
* sentence-transformers
* flask (for the web service)
### Running Instructions
* clone from new-flask-server branch
* run command: pip install -r requirements.txt in directory
* ensure ngrok is installed from [here](https://ngrok.com/download/windows)
* create venv via command: python3 -m venv venv then run venv via command: venv\Scripts\activate
* run server in venv via command: flask --app app.py run 
* once running server in venv, run ngrok then run command: ngrok http 5000
* access frontend for site [here](https://cs547-frontend-git-vercel-react-a1823d-edwardfantasias-projects.vercel.app/?_vercel_share=p6j9AqGMNUy6kSd0iSlmwYEyXmBZcfev)
* must then put in forwarding address into the top input box then click save in order to run this application
* now you can enjoy listening to the recommended music!
