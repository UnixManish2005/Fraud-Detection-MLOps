# 🛡️ Credit Card Fraud Detection System

A **production-ready**, end-to-end machine learning system for detecting credit card fraud in real time.

---

## 📁 Project Structure

```
fraud_detection/
├── data/                        # ← Place creditcard.csv here
├── notebooks/
│   └── 01_eda.ipynb             # Exploratory data analysis
├── src/
│   ├── data_preprocessing.py    # Loading, cleaning, scaling, SMOTE, splitting
│   ├── feature_engineering.py   # Additional feature utilities
│   ├── model_training.py        # Train LR / RF / XGBoost, select best
│   ├── evaluate.py              # Metrics, ROC, PR curves, SHAP
│   └── predict.py               # Inference engine (used by API + CLI)
├── models/                      # Saved model artefacts (auto-created)
│   ├── best_model.joblib
│   ├── model_meta.joblib
│   └── scaler.joblib
├── app/
│   └── main.py                  # FastAPI REST API
├── logs/                        # Training and API logs (auto-created)
├── plots/                       # Evaluation plots (auto-created)
├── train.py                     # Convenience CLI entry point
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.10 or 3.11
- pip

### Steps

```bash
# 1. Clone / download the project
cd fraud_detection

# 2. (Optional but recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 📥 Dataset Setup

Download the **Credit Card Fraud Detection** dataset from Kaggle:
👉 https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place the CSV inside the `data/` directory:

```
fraud_detection/
└── data/
    └── creditcard.csv     ← here
```

---

## 🚀 Training the Model

### Quick start (recommended defaults)

```bash
python train.py
```

### All options

```bash
python train.py --help

# Custom CSV path
python train.py --data data/creditcard.csv

# Disable SMOTE (use class_weight='balanced' instead)
python train.py --no-smote

# Tune decision threshold for max F2 score (recall-heavy)
python train.py --tune-threshold

# Train AND run full evaluation with plots
python train.py --evaluate
```

After training you will see:
```
── Model Comparison (ROC-AUC) ───────────────────────────────
  XGBoost                    AUC = 0.9821  ← best
  RandomForest               AUC = 0.9798
  LogisticRegression         AUC = 0.9742

✅  Done!  Model saved to models/best_model.joblib
```

---

## 📊 Evaluation

Run standalone evaluation (requires a trained model):

```bash
python src/evaluate.py --data data/creditcard.csv
```

Plots are saved to `plots/`:
| File | Description |
|---|---|
| `confusion_matrix_*.png` | Confusion matrix heatmap |
| `roc_curve_*.png` | ROC curve with AUC score |
| `pr_curve_*.png` | Precision-Recall curve |
| `shap_summary_*.png` | SHAP feature importance (bar) |
| `shap_beeswarm_*.png` | SHAP beeswarm plot |

---

## 🌐 Running the API

### Local

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open the **interactive docs** at:
- Swagger UI: http://localhost:8000/docs
- ReDoc:       http://localhost:8000/redoc

### Example API Requests

#### Single prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 406,
    "V1": -1.3598071, "V2": -0.0727812, "V3":  2.5363467,
    "V4":  1.3781553, "V5": -0.3383208, "V6":  0.4623878,
    "V7":  0.2395986, "V8":  0.0986980, "V9":  0.3637870,
    "V10": 0.0907942, "V11": -0.5515995, "V12": -0.6178009,
    "V13": -0.9913898, "V14": -0.3111694, "V15": 1.4681770,
    "V16": -0.4704005, "V17": 0.2079708, "V18": 0.0257906,
    "V19": 0.4039936, "V20": 0.2514121, "V21": -0.0183067,
    "V22": 0.2778376, "V23": -0.1104739, "V24": 0.0669281,
    "V25": 0.1285394, "V26": -0.1891148, "V27": 0.1335584,
    "V28": -0.0210530,
    "Amount": 149.62
  }'
```

**Response:**
```json
{
  "fraud_probability": 0.003721,
  "is_fraud": false,
  "model_name": "XGBoost",
  "threshold": 0.5,
  "latency_ms": 4.12
}
```

#### Health check

```bash
curl http://localhost:8000/health
```

#### Batch prediction

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"transactions": [{ ...transaction1... }, { ...transaction2... }]}'
```

---

## 🐳 Docker

### Build

```bash
docker build -t fraud-app .
```

### Run

```bash
docker run -p 8000:8000 fraud-app
```

> **Note:** The models directory must exist with trained models.
> Either copy your trained `models/` folder into the build context before `docker build`,
> or mount it at runtime:
> ```bash
> docker run -p 8000:8000 -v $(pwd)/models:/app/models fraud-app
> ```

---

## 🔍 CLI Prediction

Predict from the command line without starting the server:

```bash
python src/predict.py --input '{
  "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
  "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10,
  "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
  "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
  "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
  "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
  "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02,
  "Amount": 149.62
}'
```

---

## 🧠 Model Details

| Model | Strategy |
|---|---|
| **Logistic Regression** | `class_weight='balanced'`, L2, lbfgs solver |
| **Random Forest** | 300 trees, `class_weight='balanced'` |
| **XGBoost** | 300 estimators, `scale_pos_weight=100` for imbalance |

**Class imbalance handling:**
- Training: SMOTE oversampling (default) or `class_weight='balanced'`
- Selection: Model with highest ROC-AUC on held-out test set

**Threshold tuning** (optional, `--tune-threshold`):
- Maximises F2-score (weights recall 2× over precision)
- Critical for fraud: missing fraud is more costly than a false alarm

---

## 📏 Evaluation Metrics

| Metric | Why it matters for fraud |
|---|---|
| **Precision** | Of flagged transactions, how many are actually fraud |
| **Recall** | Of all real fraud, how many did we catch |
| **F1 / F2** | Balance (F2 emphasises recall) |
| **ROC-AUC** | Overall discrimination power |
| **Avg Precision** | Area under Precision-Recall curve |

---

## 📦 Dependencies

```
pandas, numpy, scikit-learn, xgboost, imbalanced-learn,
shap, fastapi, uvicorn, matplotlib, seaborn, joblib, pydantic
```

Install all with: `pip install -r requirements.txt`

---

## 📝 Logs

- **Training log:** `logs/training.log`
- **API log:** `logs/api.log`

---

## 🗺️ Roadmap / Bonus Features

- [x] Threshold tuning (`--tune-threshold`)
- [x] Structured logging to file
- [x] CLI training command (`python train.py`)
- [x] Batch prediction endpoint
- [x] Docker multi-stage build
- [x] SHAP beeswarm + bar plots
- [ ] MLflow experiment tracking
- [ ] Prometheus metrics endpoint
- [ ] Drift detection (evidently AI)
