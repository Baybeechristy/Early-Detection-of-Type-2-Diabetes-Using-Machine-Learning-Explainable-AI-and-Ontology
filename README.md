# 🏥 Early Detection of Type 2 Diabetes Using Machine Learning, Explainable AI & Clinical Rule Validation

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://2aaad6bepdxuhswjdxz9b5.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Tuned-green)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-red)
![LIME](https://img.shields.io/badge/LIME-Explainability-yellow)
![License](https://img.shields.io/badge/License-Educational-lightgrey)

> **An end-to-end ML screening pipeline that detects Type 2 Diabetes using only 5 non-invasive measurements — no blood test required.**

### 🔗 [Launch the Live Dashboard](https://2aaad6bepdxuhswjdxz9b5.streamlit.app)

---

## 📌 The Problem

- **537 million** adults worldwide have diabetes (IDF, 2021)
- **1 in 3** are diagnosed late, after complications have already developed
- Existing ML studies use lab values (HbA1c, glucose) as features — **but these are the diagnosis itself**, making prediction circular
- No reviewed study combines calibration, dual explainability, robustness validation AND clinical safety checks in one pipeline

## 💡 The Solution

A screening system that asks a harder, more useful question:

> *Can diabetes risk be identified from measurements that don't need a blood test?*

Using only **age, sex, BMI, systolic BP and diastolic BP**, this system screens patients, explains its reasoning, and catches its own mistakes through a clinical contradiction layer.

---

## 🏆 Key Results

| Metric | Value | Why It Matters |
|--------|-------|----------------|
| **Primary Model** | XGBoost (Tuned via 192-combination GridSearchCV) | Systematic optimisation, not default settings |
| **AUC** | 0.766 | Competitive for 5 non-invasive features |
| **Recall @ 0.15** | 81.3% — catches 256/315 diabetic patients | Screening priority: minimise missed cases |
| **Calibration (ECE)** | 0.0085 | Predicted probabilities match reality within 1% |
| **Robustness** | AUC std = 0.0134 across 10 seeds | Not a lucky split — genuinely stable |
| **Missed Cases** | 59 (vs 106 with default XGBoost, vs 315 at threshold 0.50) | Tuning + threshold optimisation saved 256 patients |

### 🔬 The Calibration Surprise

Post-hoc calibration (Platt Scaling, Isotonic Regression) **made things worse**, not better. The shallow XGBoost (depth=3, lr=0.01) was already naturally well-calibrated. This is a finding worth publishing — most studies assume calibration always helps.

### 🚨 The Threshold Discovery

At the default threshold of 0.50, the model catches **zero** diabetic patients. Sensitivity = 0.000. Every single diabetic patient is missed. This is not a bug — it's what happens when 15.5% prevalence pushes all probabilities below 0.50.

---

## 🧠 System Architecture

```
NHANES 2015-2018 Data (10,168 adults)
    │
    ▼
┌─────────────────────────────┐
│  Data Preprocessing         │
│  • Cycle filter (2015-18)   │
│  • ADA label construction   │
│  • BP averaging             │
│  • Age ≥ 18 filter          │
│  • 80/20 stratified split   │
└──────────┬──────────────────┘
           ▼
┌─────────────────────────────┐
│  5 Models Trained & Compared│
│  • Logistic Regression      │
│  • Random Forest            │
│  • XGBoost Default          │
│  • XGBoost Tuned ★          │
│  • Neural Network (MLP)     │
└──────────┬──────────────────┘
           ▼
┌─────────────────────────────────────────────┐
│  Evaluation Pipeline                        │
│  ┌───────────┐ ┌───────────┐ ┌────────────┐│
│  │Calibration│ │ Threshold │ │ Robustness ││
│  │ECE: 0.0085│ │ 0.15 → 81%│ │ 10 seeds   ││
│  └───────────┘ └───────────┘ └────────────┘│
└──────────┬──────────────────────────────────┘
           ▼
┌─────────────────────────────────────────────┐
│  Explainability                             │
│  ┌──────────────┐  ┌──────────────────────┐ │
│  │ SHAP         │  │ LIME                 │ │
│  │ (TreeSHAP)   │  │ (500 perturbations)  │ │
│  │ Global+Local │  │ 4 clinical cases     │ │
│  └──────────────┘  └──────────────────────┘ │
└──────────┬──────────────────────────────────┘
           ▼
┌─────────────────────────────────────────────┐
│  🚨 Clinical Rule Contradiction Layer       │
│  Compares ML prediction vs ADA criteria     │
│  4 outcomes: Agree Low / Agree High /       │
│              Model Flag / CONTRADICTION     │
└──────────┬──────────────────────────────────┘
           ▼
┌─────────────────────────────────────────────┐
│  📊 Streamlit Dashboard (8 Tabs)            │
│  Real-time screening • SHAP explanations    │
│  Interactive contradiction checker          │
└─────────────────────────────────────────────┘
```

---

## 📊 Dashboard — 8 Interactive Tabs

| Tab | What It Shows |
|-----|---------------|
| 🏠 **Home** | Dataset overview, KPIs, distributions, correlations |
| 🔍 **Screening** | Enter patient measurements → risk score + SHAP explanation + clinical rule check |
| 📈 **Model Results** | 5-model comparison, ROC curves, confusion matrices |
| 📐 **Calibration** | Brier score, ECE, calibration curves — proof that probabilities are trustworthy |
| 🎯 **Threshold** | 8 thresholds analysed — see exactly why 0.50 fails and 0.15 works |
| 🔁 **Robustness** | 10-seed validation — proof results aren't a fluke |
| 🧠 **Explainability** | Global SHAP + individual SHAP/LIME side-by-side for any patient |
| 🚨 **Clinical Rules** | 4 real NHANES case studies + interactive contradiction checker |

---

## 🚨 Clinical Rule Layer — The Novel Contribution

The most distinctive component. No reviewed study included this.

**The problem:** A lean patient with normal blood pressure can look low-risk on non-invasive features — but have severely elevated blood sugar.

**Real example from the dataset:**

| Feature | Value |
|---------|-------|
| Age | 56 |
| BMI | 24.4 (lean) |
| Blood Pressure | 114/81 (normal) |
| **Model Prediction** | **14.2% → Low Risk** |
| HbA1c | **13.7%** (threshold: 6.5%) |
| Glucose | **397 mg/dL** (threshold: 126) |
| **Clinical Rule** | **🚨 CONTRADICTION WARNING** |

Without the rule layer, this patient walks away with a clean bill of health. With it, they get flagged for immediate follow-up.

---

## 🔄 Comparison with Existing Literature

| Study | Features | AUC | Calibration | XAI | Robustness | Dashboard | Clinical Rules |
|-------|----------|-----|-------------|-----|------------|-----------|----------------|
| Kopitar (2020) | Routine clinical | 0.77–0.84 | ❌ | ❌ | ❌ | ❌ | ❌ |
| Dinh (2019) | Lab + lifestyle | 0.83 | ❌ | ❌ | ❌ | ❌ | ❌ |
| Qin (2022) | Lab + lifestyle | 0.83 | ❌ | ❌ | ❌ | ❌ | ❌ |
| Zou (2018) | Lab + clinical | 0.82 | ❌ | ❌ | ❌ | ❌ | ❌ |
| Riveros Perez (2025) | Lifestyle | 0.817 | ❌ | ❌ | ❌ | ❌ | ❌ |
| **This Project** | **5 non-invasive** | **0.766** | ** ECE 0.0085** | **SHAP+LIME** | **10 seeds** | **8 tabs** | **4 outcomes** |

Lower AUC — but that's the cost of using no blood test features. On every other dimension, this project exceeds the published literature.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| pandas / NumPy | Data processing |
| scikit-learn | ML models, evaluation, calibration |
| XGBoost | Primary model (gradient boosting) |
| SHAP | Global + local explainability (TreeExplainer) |
| LIME | Local explainability (perturbation-based) |
| Matplotlib / Seaborn | Visualisation |
| Streamlit | Interactive dashboard |

---

## 📂 Project Structure

```
├── app.py                          # Streamlit dashboard (1,800+ lines)
├── nhanes_diabetes_clean.csv       # Cleaned NHANES dataset (10,168 adults)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── Early Detection of Type 2       # Full Kaggle notebook (63 cells)
    Diabetes using ML, AI/
```

---

## 🚀 Run Locally

```bash
git clone https://github.com/Baybeechristy/Early-Detection-of-Type-2-Diabetes-Using-Machine-Learning-Explainable-AI-and-Clinicial-Reasoning.git
cd Early-Detection-of-Type-2-Diabetes-Using-Machine-Learning-Explainable-AI-and-Clinicial-Reasoning
pip install -r requirements.txt
streamlit run app.py
```

---

## 📋 Dataset

| Property | Value |
|----------|-------|
| **Source** | NHANES 2015–2018 (CDC) |
| **Size** | 10,168 adults after cleaning |
| **Prevalence** | 15.5% diabetic (1,577 cases) |
| **Label** | ADA criteria: HbA1c ≥ 6.5% or Fasting Glucose ≥ 126 mg/dL |
| **Features** | Age, Sex, BMI, Systolic BP, Diastolic BP |

---

## 👤 Author

**Irene Christabel Ogbomo**
BSc Computing — Nottingham Trent University (2025/26)
Supervised by Owa Kayode

📧 Contact: [via GitHub](https://github.com/Baybeechristy)

---

## ⚠️ Disclaimer

This system is a **research prototype** developed as a university final year project. It is **not a medical device** and must not be used for clinical diagnosis. Always consult a qualified healthcare professional for medical advice.

---
