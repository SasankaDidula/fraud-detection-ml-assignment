# Fraud Detection in Financial Transactions

Machine learning project on the [Kaggle Fraudulent Transactions dataset](https://www.kaggle.com/datasets/chitwanmanchanda/fraudulent-transactions-data).  
Implements **Logistic Regression** and **Random Forest** (non–deep learning) for classifying fraudulent financial transactions in a highly imbalanced dataset.

---

## 📌 Project Overview
- Dataset: ~6.3M rows, 11 features (`step`, `type`, `amount`, balances, etc.), with severe class imbalance (`isFraud` < 0.5%).  
- Goal: Detect fraudulent transactions while handling imbalance and preventing data leakage.  
- Models:
  - Logistic Regression → interpretable baseline with `class_weight="balanced"`.  
  - Random Forest → nonlinear, robust, provides feature importance.  
- Evaluation:
  - ROC-AUC, PR-AUC (suited for imbalance)  
  - Confusion Matrix, Precision, Recall, F1  
  - Threshold tuning to analyze precision/recall trade-offs.  

---

## 🛠️ Features
- **Preprocessing**: One-hot encoding, scaling, memory-optimized CSV loading.  
- **Feature Engineering**: Balance deltas (`tx_delta_orig`, `tx_delta_dest`).  
- **Time-Aware Split**: Train on early `step` values, test on later steps to avoid leakage.  
- **Imbalance Handling**: `class_weight` strategy for fair learning.  
- **Hardware Friendly**: `QUICK_EXPERIMENT` flag to downsample majority class (safe for 8GB RAM).  

---

## 📂 Repository Structure
```
fraud-detection-ml-assignment/
│
├── Fraud_Detection_Assignment.ipynb     # Main notebook
├── Fraud_Detection_Assignment_source.txt # Exported source for appendix
├── requirements.txt                     # Dependencies
│
├── report/
│   └── Fraud_Detection_Report.pdf       # Final report
│
├── submission/
│   ├── members.txt
│   └── submission.txt
│
└── data/                                # (ignored, store Fraud.csv here)
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/fraud-detection-ml-assignment.git
cd fraud-detection-ml-assignment
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
- Get **Fraud.csv** from Kaggle:  
  [Fraudulent Transactions Dataset](https://www.kaggle.com/datasets/chitwanmanchanda/fraudulent-transactions-data)  
- Place it in the `data/` folder.  
- Or let the notebook auto-download using the Kaggle API.

### 4. Run the notebook
Open Jupyter and run:
```bash
jupyter notebook Fraud_Detection_Assignment.ipynb
```

---

## 📊 Results
- **Logistic Regression**: Strong baseline, interpretable.  
- **Random Forest**: Better PR-AUC, highlights key transaction features.  
- Both evaluated with precision/recall trade-offs to balance fraud detection vs false positives.  

---

## 📎 Submission Files
- `members.txt` → group member details  
- `submission.txt` → dataset link, repo link, YouTube video link  

---

## 👤 Author
- **N.G.S.D. Nanayakkara (MS25948592)** – Sri Lanka Institute of Information Technology (SLIIT)  
- MSc in Artificial Intelligence  

---
