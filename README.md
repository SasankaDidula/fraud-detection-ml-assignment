# Fraud Detection in Financial Transactions (Non-Deep Learning)

## Overview
This repository contains a **time-aware, leakage-safe** fraud detection pipeline using **Logistic Regression** (non-deep learning). It demonstrates industry best practices for tabular ML:
- **Temporal split** by `step` to prevent look-ahead leakage
- **scikit-learn Pipeline** with **ColumnTransformer** (scaling + robust `OneHotEncoder(handle_unknown="ignore")`)
- **Imbalance-aware** training (class_weight="balanced")
- **Threshold selection** using Fβ and operating constraints (recall floors, alert-rate caps)
- **Evaluation** using **PR-AUC**, **ROC-AUC**, **Confusion Matrices** (raw & normalized), and PR/ROC curves
- **Reproducibility** (global seeds + `random_state`)

A second model (Random Forest) will be added by a teammate using the **same preprocessor Pipeline**.

## Project Structure
```
.
├── Fraud_Detection_Assignment.ipynb   # Main notebook
├── README.md                          # This file
├── requirements.txt                   # Reproducible environment
├── members.txt                        # Team members (IDs, names, emails)
├── submission.txt                     # Dataset + GitHub + YouTube links
├── report/
│   ├── Fraud_Detection_Report.md      # Report source (convert to PDF)
│   └── appendix/
│       └── Fraud_Detection_Assignment_source.txt  # Notebook exported as text (code appendix)
```

## How to Run
1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate    # Windows
   # source .venv/bin/activate # Linux/Mac
   pip install -r requirements.txt
   ```
2. Launch Jupyter and open `Fraud_Detection_Assignment.ipynb`:
   ```bash
   jupyter lab
   ```
3. Execute cells top-to-bottom. Ensure the dataset CSV is available or download via the link in `submission.txt`.

## Reproducibility
- All models use `random_state=42`. Set `PYTHONHASHSEED=0` if you want absolute determinism.
- Temporal split is performed by `time_aware_split(...)` with hard assertions that prove **train ≤ cutoff** and **test > cutoff(+gap)**.

## Notes
- For thresholding policy, we recommend **Recall ≥ 0.95** then maximize **Precision**, subject to your alert budget.
- To add Random Forest, reuse the same `preprocess` transformer and swap the classifier.

## License
Educational use for coursework submission; dataset license per original provider.
