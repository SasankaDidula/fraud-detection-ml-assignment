# Fraud Detection in Financial Transactions — Report

## 1. Problem Statement
Detect fraudulent transactions in a highly imbalanced financial dataset **without deep learning**, optimizing an operating threshold to achieve high recall with manageable alert volume. Emphasis is placed on **temporal validity** and **leakage prevention** to ensure deployable performance.

## 2. Dataset Description
- **Source:** PaySim synthetic financial transactions (Kaggle).
- **Key Columns:** `step` (time step), `type`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`, `isFraud`.
- **Class Imbalance:** Fraud is rare (<< 1%), requiring recall-focused evaluation (PR-AUC over ROC-AUC for decisioning).

## 3. Methodology
### 3.1 Leakage Controls
- **Temporal split** by `step`: Train `step ≤ cutoff`, Test `step > cutoff (+ gap)`.
- Hard **assertions** to prove temporal purity after splitting.

### 3.2 Features & Engineering
- Engineered deltas (`tx_delta_orig`, `tx_delta_dest`), ratios (`orig_balance_ratio`, `dest_balance_ratio`), flags (negative/no-change), and `log_amount`.
- High-cardinality identifiers dropped.

### 3.3 Modeling
- **Pipeline:** `ColumnTransformer` → `StandardScaler` (numeric) + `OneHotEncoder(handle_unknown="ignore")` (categorical) → **LogisticRegression(class_weight="balanced")`.
- **Thresholding:** Fβ (β≈2) search and/or constraints (Recall ≥ 0.95), with alert-rate awareness.

### 3.4 Evaluation
- **Metrics:** PR-AUC, ROC-AUC, Precision/Recall/F1 at chosen τ.
- **Visuals:** PR/ROC curves; Confusion Matrices (raw, normalized).
- **Reporting:** Single summary table of selected τ with CM counts and rates.

## 4. Results & Discussion
- Provide PR-AUC, ROC-AUC, chosen τ, and the trade-off rationale (recall vs precision vs alerts).
- Comment on drift risk and re-tuning cadence (e.g., monthly threshold review).

## 5. Appendix — Full Source Code (Text)
See `appendix/Fraud_Detection_Assignment_source.txt` for the full notebook source as text.
