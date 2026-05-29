<div align="center">

# 🛡️ Credit Card Fraud Detection

### Real-time fraud scoring powered by XGBoost · FastAPI · Interactive Dashboard

[![Live Demo](https://img.shields.io/badge/🔴_Live_Demo-Render-4353FF?style=for-the-badge)](https://fraud-detection-api-edgq.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-FF6600?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

> Trained on **8.9M transactions** · **0.15% fraud rate** · **36 engineered features** · **5 models compared**

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Model Results](#-model-results)
- [Project Structure](#-project-structure)
- [Features Engineered](#-features-engineered)
- [API Usage](#-api-usage)
- [Run Locally](#-run-locally)
- [Dataset](#-dataset)
- [Tech Stack](#-tech-stack)

---

## 🔍 Overview

This project builds and compares **five supervised machine learning models** for detecting fraudulent credit card transactions on the IBM synthetic dataset, then deploys the best model as a **production REST API** with an interactive web dashboard.

**Key highlights:**
- ✅ **Leakage-free pipeline** — causal velocity features, stratified train/val/test split
- ✅ **PR-AUC as primary metric** — correct choice for extreme class imbalance (0.15% fraud)
- ✅ **Explainable results** — per-transaction risk reasons (not just a probability score)
- ✅ **Production deployment** — live on Render with zero cold-start model loading

---

## 🌐 Live Demo

**👉 [https://fraud-detection-api-edgq.onrender.com](https://fraud-detection-api-edgq.onrender.com)**

| Field | Fraud Example |
|---|---|
| Amount | `$3,500` |
| Card Type | `Debit` |
| Tx Count 24h | `15` |
| Amount Sum 24h | `$9,000` |
| User Mean Amt | `$45` |
| User Std Amt | `$30` |

> Enter these values in the dashboard to see a **🚨 FRAUDULENT — BLOCK** result with ~90% confidence.

---

## 📊 Model Results

> Evaluated on a held-out test set (15% stratified). **PR-AUC** is the primary metric for imbalanced fraud detection — ROC-AUC is reported for completeness only.

| Rank | Model | PR-AUC ⭐ | ROC-AUC | F1 | Precision | Recall | Train Time |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 | **XGBoost** | **0.8901** | **0.9966** | **0.8402** | 0.8485 | 0.8320 | 16s |
| 🥈 | LightGBM | 0.7332 | 0.9902 | 0.5716 | 0.4485 | 0.7880 | 123s |
| 🥉 | MLP (Neural Net) | 0.5422 | 0.9767 | 0.0574 | 0.0297 | 0.9040 | 1522s |
| 4️⃣ | Random Forest | 0.2773 | 0.9761 | 0.1889 | 0.1078 | 0.7630 | 261s |
| 5️⃣ | Logistic Regression | 0.0279 | 0.8882 | 0.0128 | 0.0064 | 0.7880 | 20s |

**Key finding:** Gradient boosting (XGBoost, LightGBM) significantly outperforms bagging and linear methods on tabular fraud data — consistent with Grinsztajn et al. (2022) and industry practice at Stripe, PayPal, and Square.

---

## 📁 Project Structure

```
fraud-detection-api/
│
├── 📓 notebooks/
│   ├── 01_eda.ipynb                  # Exploratory data analysis & visualisation
│   ├── 02_preprocessing_v2.ipynb     # Causal feature engineering (leakage-free)
│   ├── 03_training_v2.ipynb          # Model training, Optuna tuning, evaluation
│   └── 04_evaluation.ipynb           # Charts, confusion matrices, AUC comparison
│
├── ⚡ api/
│   ├── main.py                       # FastAPI app — serves dashboard + /predict
│   ├── predictor.py                  # Feature builder & XGBoost inference
│   └── schemas.py                    # Pydantic v2 request/response models
│
├── 🤖 model/
│   ├── model.pkl                     # Trained XGBoost model (deployed)
│   └── encoders.pkl                  # LabelEncoders for card brand & state
│
├── ⚙️ config/
│   └── model_metadata.json           # Feature order, threshold, all metrics
│
├── 🎨 static/
│   └── index.html                    # Interactive dashboard (Chart.js + Vanilla JS)
│
├── 📈 outputs/                       # Generated evaluation charts & figures
├── 📋 report/                        # Model comparison CSV results
├── requirements_api.txt              # API-only dependencies (slim)
└── render.yaml                       # Render deployment configuration
```

---

## 🔧 Features Engineered (36 total)

<details>
<summary><b>Click to expand all features</b></summary>

| Category | Feature | Description |
|---|---|---|
| ⏰ **Time** | `tx_hour`, `tx_day`, `tx_month` | Transaction timestamp components |
| ⏰ **Time** | `tx_dayofweek`, `tx_is_weekend` | Day of week signal |
| ⏰ **Time** | `tx_is_night` | Flag for 10 PM – 5 AM transactions |
| 💰 **Amount** | `amount_abs`, `amount_log` | Raw and log-transformed amount |
| 💰 **Amount** | `amount_to_limit_ratio` | Amount as fraction of credit limit |
| 💰 **Amount** | `amount_to_income_ratio` | Amount as fraction of yearly income |
| 💰 **Amount** | `amount_vs_user_mean` | Deviation from user's historical average |
| 🚀 **Velocity** | `tx_count_24h`, `amount_sum_24h` | 24-hour transaction frequency & volume |
| 🚀 **Velocity** | `tx_count_7d`, `amount_sum_7d` | 7-day transaction frequency & volume |
| 🚀 **Velocity** | `time_since_last_tx` | Hours since previous transaction |
| 🚀 **Velocity** | `user_mean_amt`, `user_std_amt` | Causal per-user spending statistics |
| 👤 **User** | `age_at_tx`, `years_to_retirement` | Demographic signals |
| 👤 **User** | `debt_to_income`, `credit_score` | Financial health indicators |
| 👤 **User** | `yearly_income`, `total_debt` | Income & debt |
| 💳 **Card** | `card_type_enc`, `card_brand_enc` | Credit/Debit, Visa/MC/Amex/Discover |
| 💳 **Card** | `has_chip_flag`, `use_chip_enc` | EMV chip presence & transaction method |
| 💳 **Card** | `num_cards_issued`, `num_credit_cards` | Card count signals |
| 🏪 **Merchant** | `merchant_state_enc` | Encoded US state |
| 🏪 **Merchant** | `mcc_enc` | Merchant Category Code |

</details>

---

## 🚀 API Usage

### Health Check
```bash
curl https://fraud-detection-api-edgq.onrender.com/health
```
```json
{"status": "ok", "model": "XGBoost", "threshold": 0.5, "features": 36}
```

### Predict — Fraud Example
```bash
curl -X POST https://fraud-detection-api-edgq.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 3500,
    "use_chip": "Online Transaction",
    "mcc": 5732,
    "merchant_state": "CA",
    "timestamp": "2026-05-29T03:15:00",
    "birth_year": 1990, "current_age": 35, "retirement_age": 65,
    "gender": "Male", "yearly_income": 30000, "total_debt": 28000,
    "credit_limit": 1500, "credit_score": 450,
    "num_credit_cards": 6, "per_capita_income": 15000,
    "card_type": "Debit", "card_brand": "Visa", "has_chip": "NO",
    "time_since_last_tx": 0.05,
    "tx_count_24h": 15, "amount_sum_24h": 9000,
    "tx_count_7d": 50, "amount_sum_7d": 22000,
    "user_mean_amt": 45, "user_std_amt": 30
  }'
```

### Response
```json
{
  "fraud_probability": 0.9094,
  "is_fraud": true,
  "threshold": 0.5,
  "model": "XGBoost",
  "risk_level": "HIGH",
  "reasons": [
    {"text": "Transaction at 03:15 AM — unusual night-time hours", "flag": true},
    {"text": "Amount $3,500.00 is well above your usual spending (avg $45.00)", "flag": true},
    {"text": "High transaction frequency: 15 transactions in last 24 hours", "flag": true}
  ]
}
```

---

## 💻 Run Locally

```bash
# 1. Clone
git clone https://github.com/sokpheng72/fraud-detection-api.git
cd fraud-detection-api

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac / Linux

# 3. Install dependencies
pip install -r requirements_api.txt

# 4. Start server
uvicorn api.main:app --port 8002 --reload

# 5. Open browser → http://localhost:8002
```

---

## 📦 Dataset

**IBM Synthetic Credit Card Transactions**
- 8,914,963 labeled transactions
- 2,000 simulated users · 2010–2019
- Fraud rate: **0.15%** (severely imbalanced)
- Source: [Kaggle — IBM Credit Card Transactions](https://www.kaggle.com/)

> Raw data files are excluded from this repository due to size. See `.gitignore` for details.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| 🤖 Model | XGBoost 2.0.3 |
| ⚡ API | FastAPI 0.111.0 + Uvicorn |
| ✅ Validation | Pydantic v2 |
| 🎨 Frontend | Vanilla JS + Chart.js |
| ☁️ Deployment | Render (free tier) |
| 🐍 Python | 3.11 |

---

## 📄 Research Context

This project is part of an undergraduate thesis comparing supervised learning approaches for real-time credit card fraud detection on highly imbalanced tabular data.

**Thesis findings:**
1. Gradient boosting dominates on tabular fraud data (XGBoost PR-AUC 0.89 vs MLP 0.54)
2. Causal velocity features are the most important signal for fraud detection
3. PR-AUC, not ROC-AUC, is the correct evaluation metric under extreme class imbalance
4. Threshold tuning on a held-out validation set is essential for deployable recall

---

<div align="center">

Made with ❤️ for research · Deployed on Render · Dataset by IBM

</div>
