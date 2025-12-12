# Crack Detection App (Monorepo)

This repository follows a professional ML engineering structure, separating research (notebooks), data handling, model artifacts, and production code (src).

## Project Structure

```text
App-Crack/
├── data/                   # Data storage (Ignored by Git)
│   ├── raw/                # Original immutable data
│   └── processed/          # Preprocessed data for training
│
├── models/                 # Model artifacts
│   ├── production/         # Active model (ONNX) used by the app
│   └── archive/            # Archived model versions
│
├── notebooks/              # Jupyter Notebooks for experiments
│
├── src/                    # Source Code
│   ├── backend/            # FastAPI Application
│   └── frontend/           # React/Vite Application
│
├── scripts/                # Utility scripts
├── tests/                  # Unit and Integration tests
└── docker-compose.yml      # Container orchestration
```

## Setup & Run

### Backend
```bash
cd src/backend
# Activate venv if needed
python app.py
```

### Frontend
```bash
cd src/frontend
npm install
npm run dev
```
