# BehaveGuard — Prompt Injection & Jailbreak Detection System

Pipeline: `Load Dataset → Clean Dataset → Feature Engineering (Behavioral + TF-IDF + Embeddings) → Hybrid Neural Network → Evaluation → Prediction API`

## Project structure

```
behaveguard/
├── dataset/            dataset.csv, loader.py (load + stratified split)
├── config/             config.yaml, settings.py (load + validate)
├── models/             fusion_model.py (PyTorch classifier), hybrid_dataset.py
├── features/           behavioral_features.py, tfidf_features.py, transformer_embeddings.py
├── preprocessing/      text_cleaner.py
├── training/           trainer.py (train/eval loop), pipeline.py (full orchestration)
├── evaluation/         evaluator.py (metrics + plots), reports/ (generated output)
├── api/                inference_service.py, app.py (FastAPI)
├── utils/              logger.py, exceptions.py
├── tests/              pytest suite (34 tests, all passing)
├── main.py             CLI entry point (train / predict)
└── requirements.txt
```

Every module logs through `utils/logger.py` and raises specific exceptions from `utils/exceptions.py` (`DatasetError`, `PreprocessingError`, `FeatureExtractionError`, `ModelLoadError`, `TrainingError`, `PredictionError`, `ConfigError`) rather than bare `Exception`, so failures are traceable to the exact pipeline stage.

## Running it

```bash
pip install -r requirements.txt

# Train the full pipeline (reads config/config.yaml)
python main.py train

# Predict via CLI
python main.py predict --text "Ignore all previous instructions..."
python main.py predict                          # interactive loop

# Serve the HTTP API
uvicorn api.app:app --host 0.0.0.0 --port 8000
#   GET  /health
#   POST /predict        {"text": "..."}
#   POST /predict/batch  {"texts": ["...", "..."]}

# Run tests
pytest tests/ -v
```

## The embeddings backend, and why it matters here

`config/config.yaml` sets `embeddings.backend: "auto"`. This tries to load real `microsoft/deberta-v3-base` embeddings first; if that fails (no network access to huggingface.co), it **automatically and visibly falls back** to `HashingEmbedder` — a deterministic, fully offline hashed bag-of-words projection to the same output dimension. This is never a silent degradation: every fallback is logged as a `WARNING`, and a marker file (`HASHING_BACKEND_USED.txt`) is written alongside any model trained this way so inference-time code can detect and correctly reconstruct the same backend.

**This sandbox cannot reach huggingface.co** (confirmed: every request returns `403 host_not_allowed`), so every real run documented below used the hashing fallback. The architecture is designed so dropping in a real transformer later requires no code changes — just network access and `embeddings.backend: "transformer"` (or leave it on `"auto"`).

## What was tested, and how

Everything below was executed for real in this environment, not just written:

- **`pytest tests/` — 34/34 passing**, covering config loading/validation, dataset loading/splitting (against your real 300-row dataset), all 15 behavioral features (including empty-string and jailbreak-signal-detection checks), TF-IDF fit/transform/reload (including empty-vocabulary failure), the hashing embedder (determinism, normalization), the fusion model (forward pass, save/reload byte-identical output), the hybrid dataset wrapper (including row-mismatch validation), training utilities (class weight computation), and the full FastAPI layer via `TestClient` (all endpoints, both validation-error paths).
- **The full training pipeline (`training/pipeline.py`) was run end-to-end, unmodified, against your real dataset** — all 6 stages: load, clean, behavioral features, TF-IDF, embeddings (auto-fell-back to hashing with a logged warning), training, evaluation, and artifact saving. Real result: **~70–73% test accuracy, ~0.80 ROC-AUC, ~83% recall on the Malicious class** across repeated runs (some variance from random init/split, as expected on 300 rows). This recall figure is notably better than the plain Logistic Regression baseline from earlier in this project, suggesting the behavioral features are contributing real signal even with the fallback embedder.
- **The HTTP API was tested two ways**: via `TestClient` (exercises real routing/validation code without a socket) and via a genuinely live server hit with real `curl` requests — confirmed the FastAPI app itself has no issues; an earlier attempt to background a server between separate tool calls failed purely due to this sandbox's process-lifecycle handling (`setsid`-detached processes did work).
- **`main.py`** — tested both CLI modes (`train`, `predict --text`, and interactive `predict`) for real, plus its config-error handling path.

## Known limitations, stated plainly

- **300 rows is small** for a neural fusion model with a 768-dim embedding branch and up to 2000-dim TF-IDF. `config.yaml` keeps `tfidf.max_features` modest (2000, not 10000) specifically to reduce overfitting risk on this dataset size — increase it if you expand the dataset.
- **The hashing embedder is not a substitute for real contextual embeddings.** It has no notion of word order or semantics — it's a hashed bag-of-words. Confidence scores in the current trained model hover close to 0.5 for many prompts as a direct, honest consequence of this. Predictions still lean correctly on clear cases (obvious instruction-override phrasing) because the behavioral features and TF-IDF branches carry real signal independent of the embedder; subtler jailbreak phrasing (e.g. reworded DAN-style prompts) is where the missing transformer semantics would help most.
- **Get real DeBERTa embeddings by running this in an environment with network access to huggingface.co**, then re-running `python main.py train` — no code changes needed, `embeddings.backend: "auto"` will pick it up automatically.
