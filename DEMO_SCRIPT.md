# 🎬 Demo Video Script — MLOps Assignment 2

> **Duration**: < 5 minutes | **Tool**: OBS Studio or Win+G

## Pre-Recording Setup

- [ ] VS Code open with project
- [ ] MLflow UI running (`python -m mlflow ui` → localhost:5000)
- [ ] API server running (`python -m uvicorn app.main:app --reload`)
- [ ] Browser tabs: MLflow, Swagger (localhost:8000/docs), GitHub Actions
- [ ] Sample cat/dog image on Desktop

---

## ⏱️ 0:00–0:30 — Introduction (30s)

**[Show: VS Code project sidebar]**

> *"Hi, I'm Santhosh. This is my end-to-end MLOps pipeline for Cats vs Dogs classification using TensorFlow MobileNetV2, FastAPI, Docker, and GitHub Actions."*

**Show briefly**: `src/`, `app/`, `.github/workflows/`, `Dockerfile`, `docker-compose.yml`

---

## ⏱️ 0:30–1:00 — M1: Versioning (30s)

**[Terminal]**

```bash
git log --oneline -5
git lfs ls-files
cat .gitattributes
```

> *"Git tracks source code. Git LFS tracks the model.h5 file which is over 15MB."*

---

## ⏱️ 1:00–2:00 — M1: Training & MLflow (60s)

> *"The model was trained using MobileNetV2 transfer learning. All metrics are tracked in MLflow."*

**[Browser → localhost:5000]**

1. Show experiment runs table
2. Click latest run → **Metrics tab** → show `train_loss`, `val_accuracy` charts
3. **Artifacts tab** → click `charts/confusion_matrix.png`, `charts/loss_curve.png`

> *"MLflow logs per-epoch metrics with interactive charts, plus the confusion matrix and classification report as artifacts."*

---

## ⏱️ 2:00–2:45 — M2: API & Prediction (45s)

> *"The model is served via FastAPI with a health check and prediction endpoint."*

**[Browser → localhost:8000/docs]**

1. Click **POST /predict** → Try it out
2. Upload sample image → Execute
3. Show JSON response

> *"It correctly predicts [Cat/Dog] with [X]% confidence. The API also exposes Prometheus metrics at /metrics."*

---

## ⏱️ 2:45–3:30 — M3: CI Pipeline (45s)

**[VS Code → `.github/workflows/pipeline.yml`]** (briefly)

> *"GitHub Actions runs on every push — checkout with LFS, install deps, run pytest, build Docker image, and push to Docker Hub."*

**[Browser → GitHub → Actions tab]**

1. Click latest successful run
2. Expand test step → show pytest passing
3. Show Docker build/push step

> *"The image is published as `msanthoshofficial/cat-dog-classifier:latest`."*

---

## ⏱️ 3:30–4:15 — M4: Docker Deployment (45s)

**[Terminal]**

```bash
docker pull msanthoshofficial/cat-dog-classifier:latest
docker run -p 8000:8000 msanthoshofficial/cat-dog-classifier:latest
```

> *"The production image is pulled from Docker Hub and deployed. Let me run the smoke test to verify."*

```bash
python smoke_test.py
```

> *"Health check and prediction both pass — deployment is successful."*

---

## ⏱️ 4:15–4:50 — M5: Monitoring (35s)

**[Browser → localhost:8000/metrics]**

> *"Prometheus metrics track request count, latency, and status codes for every API call."*

Scroll to show `http_requests_total`, `http_request_duration_seconds`

> *"These can be scraped by a Prometheus server for dashboards and alerting."*

---

## ⏱️ 4:50–5:00 — Wrap Up (10s)

> *"This covers the full MLOps lifecycle — training, experiment tracking, containerized serving, automated CI/CD, and monitoring. Thank you!"*

---

## Quick Commands Reference

```bash
git log --oneline -5              # Versioning
git lfs ls-files                  # LFS tracking
python -m src.train               # Train model
python -m mlflow ui               # MLflow → localhost:5000
python -m uvicorn app.main:app --reload  # API → localhost:8000
docker pull msanthoshofficial/cat-dog-classifier:latest
docker run -p 8000:8000 msanthoshofficial/cat-dog-classifier:latest
python smoke_test.py              # Smoke test
```
