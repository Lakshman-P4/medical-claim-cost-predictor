# Medical Claim Cost Prediction System

## Workers' Compensation Claim Reserving via Complexity-Routed Machine Learning

A multi-stage machine learning pipeline that predicts total medical claim costs for workers' compensation insurance. The system classifies claims by predictive complexity, then routes each claim to a specialized regression model trained for that complexity tier — delivering more accurate cost estimates than any single-model approach.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Solution Overview](#solution-overview)
- [Data](#data)
- [Pipeline Architecture](#pipeline-architecture)
  - [Stage 1: Data Preprocessing](#stage-1-data-preprocessing)
  - [Stage 2: Feature Engineering](#stage-2-feature-engineering)
  - [Stage 3: LSTM Sequence Learning](#stage-3-lstm-sequence-learning)
  - [Stage 4: Complexity Router](#stage-4-complexity-router)
  - [Stage 5: Specialized Regression Models](#stage-5-specialized-regression-models)
  - [Stage 6: Inference Pipeline](#stage-6-inference-pipeline)
- [Results](#results)
- [Key Technical Decisions](#key-technical-decisions)
- [Project Structure](#project-structure)
- [Setup and Usage](#setup-and-usage)
- [Future Work](#future-work)

---

## Problem Statement

In workers' compensation insurance, setting accurate **claim reserves** (predicted total cost) is critical for financial planning, risk management, and regulatory compliance. Traditional approaches rely on adjuster experience and simple averages by injury type, which leads to:

- **Under-reserving** on complex claims, creating unexpected financial exposure
- **Over-reserving** on simple claims, tying up capital unnecessarily
- **Inconsistent estimates** across adjusters handling similar claims
- **Late identification** of high-cost claims that require senior review

The goal is to build a data-driven system that predicts total claim cost using the information available in medical visit records, enabling early and accurate reserve-setting.

---

## Solution Overview

Rather than treating all claims identically, this system recognizes that claims vary dramatically in how predictable they are. A simple sprain with two visits behaves very differently from a complex back injury with 30+ visits. Forcing a single model to handle both produces mediocre results across the board.

The solution is a **Complexity-Routed Prediction Pipeline**:

```
                        ┌─────────────────────────────┐
                        │      Raw Visit Data          │
                        │   631,847 visits / 100,130   │
                        │         claims               │
                        └──────────────┬──────────────┘
                                       │
                        ┌──────────────▼──────────────┐
                        │   Feature Engineering        │
                        │   100+ engineered features   │
                        └──────────────┬──────────────┘
                                       │
                        ┌──────────────▼──────────────┐
                        │   LSTM Sequence Learning     │
                        │   128-dim embeddings per     │
                        │   claim (autoencoder)        │
                        └──────────────┬──────────────┘
                                       │
                        ┌──────────────▼──────────────┐
                        │   XGBoost Complexity Router  │
                        │   87.17% routing accuracy    │
                        └───┬──────────┼──────────┬───┘
                            │          │          │
                    ┌───────▼──┐  ┌────▼─────┐  ┌─▼───────┐
                    │   LOW    │  │   MED    │  │  HIGH   │
                    │  Model   │  │  Model   │  │  Model  │
                    │  33,128  │  │  33,384  │  │  33,618 │
                    │  claims  │  │  claims  │  │  claims │
                    └───────┬──┘  └────┬─────┘  └─┬───────┘
                            │          │          │
                        ┌───▼──────────▼──────────▼───┐
                        │     Predicted Claim Cost     │
                        │     + Complexity Class       │
                        │     + Confidence Indicator   │
                        └─────────────────────────────┘
```

---

## Data

The dataset consists of **631,847 visit-level records** across **100,130 unique workers' compensation medical claims**.

**Key characteristics:**

| Attribute             | Detail                                            |
| --------------------- | ------------------------------------------------- |
| Rows                  | 631,847 (visit-level)                             |
| Unique Claims         | 100,130                                           |
| Features (raw)        | 23 columns                                        |
| Features (engineered) | 100+ columns                                      |
| Target Variable       | `TOTAL_CLAIM_COST` (total paid across all visits) |
| Cost Range            | $0.06 – $49,914.04                                |

**Features include:**

- **Demographics:** Age, gender, incident state
- **Claim characteristics:** Claimant type, body part, nature of injury, claim cause group
- **Medical coding:** ICD codes (primary and count), presence of ICD flag
- **Visit data:** Sequential payment amounts per visit, visit number, cumulative payment totals
- **Derived visit metrics:** Total visits, paid visits, zero/negative visits, paid visit ratio

---

## Pipeline Architecture

### Stage 1: Data Preprocessing

Raw claims data is cleaned and structured into visit-level records. Each row represents one medical visit for a claim, ordered sequentially by `NO_OF_VISIT`.

**Key operations:**

- Standardize column names and data types
- Handle missing values in ICD codes and categorical fields
- Validate sequential ordering of visits within each claim
- Compute cumulative payment totals (`MEDICAL_PAYMENT_TOTAL`)
- Calculate visit-level aggregates: `TOTAL_VISITS_TRUE`, `TOTAL_PAID_VISITS`, `TOTAL_ZERO_VISITS`, `TOTAL_NEGATIVE_VISITS`, `PAID_VISIT_RATIO`

**Output:** `df_preprocessed.csv` — 631,847 rows × 23 columns

### Stage 2: Feature Engineering

Visit-level data is enriched with 100+ engineered features designed to capture claim behavior patterns that raw fields cannot express alone.

**Feature categories:**

**First-visit features** — Available at the time of prediction:

- `FIRST_VISIT_PAYMENT`: Medical amount at visit 1
- `LOG_PAYMENT`: Log-transformed first payment
- ICD presence and primary code

**Payment pattern features** — Derived from the full visit sequence:

- `PAYMENT_TREND`: Direction of payment amounts over time
- `PAYMENT_VOLATILITY`: Standard deviation of payment changes
- `DIRECTION_CHANGES`: Number of times payment direction reverses
- `EARLY_PAYMENT_RATIO`: Proportion of total paid in first half of visits

**Context-relative features** — Measuring how a claim compares to similar claims:

- Z-scores computed within context groups defined by `[AGE_GROUP, BODY_PART_GROUP, NATURE_OF_INJURY]`
- `PAYMENT_ZSCORE_FROM_CONTEXT`, `VISITS_ZSCORE`
- Context groups yielded 925 distinct combinations

**Categorical encoding** — Frequency encoding used instead of one-hot to avoid sparse matrices:

- Each categorical value encoded as its frequency proportion
- Preserves information density for distance-based and tree-based models

**Combination rarity features** — 15 COMBO features capturing how unusual specific feature combinations are:

- Example: `COMBO_15` = `PAYMENT_BIN + ICD_BIN + INCIDENT_STATE`
- Rarity measured as inverse frequency of the combination

**Output:** `df_engineered.csv` — 100,130 claims × 100+ features

### Stage 3: LSTM Sequence Learning

A bidirectional LSTM autoencoder processes the full visit sequence for each claim, producing a dense 128-dimensional embedding that captures temporal patterns in the data.

**Why an autoencoder (not a predictor):**

An earlier iteration trained the LSTM to predict total claim cost directly. While this achieved R² = 0.76, the resulting embeddings contained information optimized for cost prediction — which constitutes data leakage when those embeddings are later used as features in the router and regression models.

The autoencoder approach trains the LSTM to **reconstruct its input sequence**, not predict the target. This means the embeddings capture genuine sequential patterns (payment trajectories, visit frequency dynamics, injury progression) without encoding the answer.

| Approach         | Objective            | Embeddings Contain       | Leakage? |
| ---------------- | -------------------- | ------------------------ | -------- |
| LSTM Predictor   | Predict total cost   | Cost prediction signal   | Yes      |
| LSTM Autoencoder | Reconstruct sequence | Sequential patterns only | No       |

**Architecture:**

```
Input: Padded visit sequence (max 50 visits × features per visit)
    → Masking layer (handles variable-length sequences)
    → Bidirectional LSTM (64 units)
    → Dense (128 units, ReLU) ← Embedding layer
    → RepeatVector
    → LSTM Decoder
    → TimeDistributed Dense → Reconstructed sequence

Loss: RMSE (reconstruction error)
Epochs: 50 (with early stopping)
```

**Embedding validation:**

The autoencoder embeddings were validated by using them alone (without engineered features) to train a router, achieving 79.55% accuracy — confirming the embeddings capture meaningful claim-level signal without leakage.

**Output:** `claim_embeddings.csv` — 100,130 claims × 128 embedding dimensions

### Stage 4: Complexity Router

The router classifies each claim into one of three complexity tiers: **LOW**, **MED**, or **HIGH**. Complexity is defined by **predictability** — how difficult a claim is to predict accurately — not by payment amount.

**Evolution of the router:**

The routing system went through several iterations, each addressing a fundamental flaw in the previous approach:

| Version                                 | Method                          | Accuracy   | Issue                                             |
| --------------------------------------- | ------------------------------- | ---------- | ------------------------------------------------- |
| v1: Hand-tuned KNN                      | Weighted scoring + KNN voting   | N/A        | KNN had 0% flip rate (rubber-stamped seed labels) |
| v2: Predictability-based KNN            | Residual-based labels + KNN     | ~63%       | Barely above random for 3-class problem           |
| v3: XGBoost (engineered features only)  | XGBoost classifier              | 70.63%     | MED accuracy only 56.61%                          |
| v4: XGBoost + LSTM embeddings           | XGBoost with 128-dim embeddings | 79.55%     | Embeddings-only, no engineered features           |
| **v5: XGBoost + embeddings + features** | **Full feature set**            | **87.17%** | **Production version**                            |

**Final router configuration:**

- Model: XGBoost classifier
- Features: 128 LSTM embedding dimensions + 100+ engineered features
- Cross-validation: 5-fold stratified
- Hyperparameter search: RandomizedSearchCV (50 iterations)

**Per-class routing accuracy:**

| Complexity  | Accuracy   | Precision | Recall | Claims      |
| ----------- | ---------- | --------- | ------ | ----------- |
| LOW         | 71.01%     | —         | —      | 33,128      |
| MED         | 76.38%     | —         | —      | 33,384      |
| HIGH        | 91.27%     | —         | —      | 33,618      |
| **Overall** | **87.17%** | —         | —      | **100,130** |

**Why this matters:**

The router does not need to be perfect. Even a moderately accurate router (80%+) improves system performance because claims routed to the wrong model still receive a reasonable prediction — the models are not wildly different. The router's main value is ensuring the _most extreme_ claims (very simple or very complex) reach the model best equipped to handle them.

### Stage 5: Specialized Regression Models

Each complexity tier has its own regression model, trained exclusively on claims of that complexity. Five model architectures were evaluated for each tier using 10-fold cross-validation:

- **XGBoost** — Gradient boosted trees
- **LightGBM** — Light gradient boosting
- **CatBoost** — Categorical boosting
- **RandomForest** — Ensemble of decision trees
- **Hybrid** — LSTM + Dense neural network combining sequence learning with tabular features

**Training approach:**

All regression models are trained on **partial visit sequences** using LSTM embeddings as features. The models learn from the complete historical visit data (all visits available for each claim) and use the embeddings to capture sequential patterns. This approach follows the principle: _train on complete data, predict with available data_.

**Results — Final Model Iteration (Actual labels, 5 models per tier):**

| Complexity | Claims | Samples | Cost Range      | Best Model | R²     | MAE    |
| ---------- | ------ | ------- | --------------- | ---------- | ------ | ------ |
| LOW        | 33,128 | 46,618  | $0.48 – $3,667  | XGBoost    | 0.6933 | $70    |
| MED        | 33,384 | 180,439 | $0.06 – $17,492 | XGBoost    | 0.6228 | $382   |
| HIGH       | 33,373 | 404,790 | $0.75 – $49,914 | XGBoost    | 0.8770 | $1,402 |

**Why HIGH claims have the highest R²:**

This is counterintuitive at first — HIGH complexity claims should be harder to predict. The explanation is data volume. HIGH complexity claims have more visits per claim (average ~18 visits vs ~2 for LOW), which means the LSTM has far more sequential data to learn from. More visits produce richer embeddings, and more partial-sequence training samples (404,790 for HIGH vs 46,618 for LOW). The LSTM's strength is learning from sequences, and longer sequences give it more to work with.

| Tier | Avg Visits | Training Samples | R²   | Explanation                    |
| ---- | ---------- | ---------------- | ---- | ------------------------------ |
| LOW  | ~2         | 46,618           | 0.69 | Few visits = limited signal    |
| MED  | ~6         | 180,439          | 0.62 | Moderate signal                |
| HIGH | ~18        | 404,790          | 0.88 | Rich sequences = strong signal |

**Note on MAPE:** MAPE scores appeared extremely high (240–2000%+) across all models. This is misleading because the dataset contains many small-dollar claims. When actual cost is $10 and predicted cost is $50, the percentage error is 400% even though the dollar error is only $40. MAE and R² are the appropriate metrics for this cost distribution.

### Stage 6: Inference Pipeline

The inference pipeline chains all trained components into a single prediction flow:

```
New visit data
    → Feature engineering (same pipeline as training)
    → LSTM autoencoder → 128-dim embedding
    → XGBoost router → Predicted complexity (LOW / MED / HIGH)
    → Route to specialized regression model
    → Predicted total claim cost (with negative clipping to $0)
```

**Outputs per claim:**

- Predicted total claim cost
- Complexity classification (LOW / MED / HIGH)
- Current amount spent (if partial visit data available)
- Predicted remaining cost
- High-cost flag (for claims exceeding configurable threshold)

---

## Results

### System Performance Summary

| Component              | Metric           | Value           |
| ---------------------- | ---------------- | --------------- |
| LSTM Sequence Learning | R² (standalone)  | 0.7647 (76.47%) |
| Complexity Router      | Overall Accuracy | 87.17%          |
| LOW Regression         | R² / MAE         | 0.69 / $70      |
| MED Regression         | R² / MAE         | 0.62 / $382     |
| HIGH Regression        | R² / MAE         | 0.88 / $1,402   |

### Error in Context

| Complexity | MAE    | Mean Claim Cost | Error as % of Mean |
| ---------- | ------ | --------------- | ------------------ |
| LOW        | $70    | ~$243           | ~29%               |
| MED        | $382   | ~$1,004         | ~38%               |
| HIGH       | $1,402 | ~$6,290         | ~22%               |

### Improvement Over Baseline

| Stage                                       | Approach          | R²                   |
| ------------------------------------------- | ----------------- | -------------------- |
| First-visit features only                   | Simple regression | 0.07 (7%)            |
| LSTM sequence learning                      | Standalone LSTM   | 0.76 (76%)           |
| Full pipeline (router + specialized models) | Complexity-routed | 0.69 – 0.88 per tier |

The LSTM sequence learning represents the single largest improvement: a **10× increase in explanatory power** from R² = 0.07 to R² = 0.76, confirming that the sequential visit pattern is the dominant signal for predicting claim costs.

---

## Key Technical Decisions

### 1. Predictability-Based Complexity (Not Cost-Based)

An early version defined complexity by payment amount (high-cost = HIGH complexity). This is circular — the router needs to predict the thing we're trying to predict. Instead, complexity is defined by how accurately a baseline model can predict the claim. Claims with small prediction errors are LOW; claims with large errors are HIGH. This focuses each specialized model on the claims that actually need specialized attention.

### 2. Data Leakage Discovery and Resolution

The original KNN router used `TOTAL_PAYMENT` (the target variable) to compute complexity features like `PAYMENT_ZSCORE_FROM_CONTEXT` and `CONTEXT_COV`. This is data leakage — using the answer to decide how to get the answer. The router was rebuilt to classify using only features available after claim completion but before needing a cost prediction.

Similarly, the first LSTM was trained as a cost predictor, meaning its embeddings encoded cost-prediction signal. Switching to an autoencoder objective (sequence reconstruction) eliminated this leakage path while preserving the sequential pattern information.

### 3. Frequency Encoding Over One-Hot

Categorical features (body part, injury type, state, ICD codes) have high cardinality. One-hot encoding created ~130 sparse columns that diluted the signal for distance-based models. Frequency encoding replaces each category with its proportion in the training data, keeping all information in a single dense column per feature.

### 4. Hybrid Model Architecture

The Hybrid model combines LSTM sequential learning with dense-layer tabular learning. While XGBoost ultimately won on the final model iteration (with actual labels), the Hybrid architecture won in the earlier iteration and demonstrated competitive performance throughout, validating the approach of combining sequence and tabular information.

---

## Project Structure

```
medical-claim-predictor/
│
├── data_preprocessed/
│   ├── df_preprocessed.csv          # Visit-level data (631,847 rows)
│   └── df_engineered.csv            # Claim-level engineered features
│
├── artifacts/
│   ├── lstm_autoencoder/
│   │   ├── autoencoder_model.h5     # Trained LSTM autoencoder
│   │   └── claim_embeddings.csv     # 128-dim embeddings per claim
│   │
│   ├── router_with_embeddings/
│   │   ├── router_xgb_best.joblib   # XGBoost complexity router
│   │   └── router_artifacts.pkl     # Scalers, encoders, metadata
│   │
│   └── models/
│       ├── low_best_model.*         # LOW complexity regression
│       ├── low_artifacts.pkl
│       ├── med_best_model.*         # MED complexity regression
│       ├── med_artifacts.pkl
│       ├── high_best_model.*        # HIGH complexity regression
│       └── high_artifacts.pkl
│
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_feature_engineering.py
│   ├── 03_lstm_autoencoder.py
│   ├── 04_complexity_router.py
│   ├── 05_regress_low.py
│   ├── 06_regress_med.py
│   ├── 07_regress_high.py
│   └── 08_final_inference.py
│
├── README.md
└── requirements.txt
```

---

## Setup and Usage

### Requirements

```
python >= 3.8
tensorflow >= 2.10
xgboost >= 1.7
lightgbm >= 3.3
catboost >= 1.1
scikit-learn >= 1.2
pandas >= 1.5
numpy >= 1.23
matplotlib >= 3.6
seaborn >= 0.12
joblib >= 1.2
```

### Installation

```bash
git clone https://github.com/<your-username>/medical-claim-predictor.git
cd medical-claim-predictor
pip install -r requirements.txt
```

### Running the Pipeline

Execute scripts in order:

```bash
# Stage 1-2: Preprocessing and feature engineering
python notebooks/01_preprocessing.ipynb
python notebooks/02_feature_engineering.py

# Stage 3: LSTM sequence learning
python notebooks/03_lstm_autoencoder.py

# Stage 4: Complexity router
python notebooks/04_complexity_router.py

# Stage 5: Regression models (one per complexity tier)
python notebooks/05_regress_low.py
python notebooks/06_regress_med.py
python notebooks/07_regress_high.py

# Stage 6: Final inference
python notebooks/08_final_inference.py
```

### Inference on New Claims

```python
from inference_pipeline import predict_claim_cost

result = predict_claim_cost(new_claim_visits_df)
# Returns: predicted_total, complexity_class, confidence
```

---

## Future Work

- **Periodic re-prediction:** Update predictions as new visits come in, improving accuracy over the claim lifecycle
- **Web application:** Streamlit-based interface for claims adjusters to input claim data and receive predictions
- **Model retraining automation:** Scheduled retraining on new claim data to maintain model accuracy over time
- **Confidence intervals:** Quantile regression or bootstrapping to provide prediction ranges alongside point estimates
- **Feature expansion:** Incorporate additional data sources (pharmacy costs, rehabilitation data, return-to-work timelines)

---

## Author

Built as a machine learning engineering project for workers' compensation claim cost prediction, demonstrating the value of complexity-routed architectures over single-model approaches in insurance reserving.
