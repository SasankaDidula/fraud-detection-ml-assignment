# ============================================================================
# FRAUD DETECTION IN FINANCIAL TRANSACTIONS - COMPLETE ANALYSIS
# ============================================================================
# Course: IT5022 - Fundamentals of Machine Learning (MSc AI)
# Team Members:
#   - [Teammate Name/ID] - Logistic Regression, Threshold Tuning
#   - MS25948592 (N.G.S.D. Nanayakkara) - Feature Engineering, Random Forest
# Dataset: Fraudulent Transactions Data (Kaggle)
# Models: Logistic Regression (baseline) & Random Forest (ensemble)
# ============================================================================

# ============================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ============================================================================

# Standard library
from pathlib import Path
import os
import gc
import zipfile
import warnings
import json
import shutil
import time

# For pretty DataFrame display in notebooks/scripts
from IPython.display import display

# Third-party core
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

# Scikit-learn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    confusion_matrix,
    RocCurveDisplay, PrecisionRecallDisplay,
    ConfusionMatrixDisplay, roc_curve, precision_recall_curve
)
from sklearn.preprocessing import MaxAbsScaler
from joblib import Memory

# Reproducibility & plotting defaults
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.grid"] = True
sns.set_style('whitegrid')

warnings.filterwarnings("ignore", category=UserWarning)

# Create figures directory
os.makedirs('figures', exist_ok=True)

print("="*80)
print("FRAUD DETECTION - LOGISTIC REGRESSION & RANDOM FOREST")
print("="*80)

# ============================================================================
# SECTION 2: CONFIGURATION & DATA LOADING
# ============================================================================

# Configuration
QUICK_EXPERIMENT = True  # Set to True for faster testing on limited RAM
MAX_ROWS = 500_000
TRAIN_CUTOFF_STEP = 600
DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)
CSV_FILENAME = "Fraud.csv"
FAST_DEV = True          # Toggle for quick iterations (True -> faster)
DS_RATIO = 20            # Keep ~20 legit per 1 fraud in TRAIN only (used when FAST_DEV=True)
LR_MAX_ITER_DEV = 200    # Faster convergence during dev
LR_TOL_DEV = 1e-3        # Looser tolerance for speed
CACHE_DIR = "sk_cache"   # Pipeline cache to avoid recomputing transforms
os.makedirs(CACHE_DIR, exist_ok=True)
memory = Memory(location=CACHE_DIR, verbose=0)

print("\nConfiguration:")
print(f"  Quick Experiment Mode: {QUICK_EXPERIMENT}")
print(f"  Max Rows (if quick): {MAX_ROWS:,}")
print(f"  Train/Test Cutoff Step: {TRAIN_CUTOFF_STEP}")

# Data loading function
def read_data_pandas(
    path: Path,
    quick_experiment: bool = False,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Load dataset with memory-efficient dtypes."""
    
    if not path.exists():
        raise FileNotFoundError(f"CSV not found at: {path}")
    
    dtype_map = {
        "step": "int32",
        "type": "category",
        "amount": "float32",
        "nameOrig": "category",
        "oldbalanceOrg": "float32",
        "newbalanceOrig": "float32",
        "nameDest": "category",
        "oldbalanceDest": "float32",
        "newbalanceDest": "float32",
        "isFraud": "int8",
        "isFlaggedFraud": "int8",
    }
    usecols = list(dtype_map.keys())
    
    read_kwargs = {
        "usecols": usecols,
        "dtype": dtype_map,
        "low_memory": False,
        "engine": "c",
    }
    if quick_experiment and max_rows is not None:
        read_kwargs["nrows"] = int(max_rows)
    
    df = pd.read_csv(path, **read_kwargs)
    
    mem_mb = df.memory_usage(deep=True).sum() / (1024**2)
    print(f"✅ Loaded {len(df):,} rows x {df.shape[1]} cols (~{mem_mb:.1f} MB)")
    
    return df

# Load data
csv_path = DATA_DIR / CSV_FILENAME

def _find_csv_after_download(data_dir: Path) -> Path | None:
    """See if a CSV exists; if multiple, pick a preferred/ largest."""
    preferred = [
        "Fraud.csv",
        "fraudTrain.csv",
        "PS_20174392719_1491204439457_log.csv"
    ]
    for name in preferred:
        p = data_dir / name
        if p.exists():
            return p
    csvs = list(data_dir.glob("*.csv"))
    if not csvs:
        return None
    return max(csvs, key=lambda p: p.stat().st_size)

def _kaggle_available() -> bool:
    return shutil.which("kaggle") is not None

def _download_from_kaggle(data_dir: Path) -> None:
    print("Attempting Kaggle download…")
    rc = os.system("kaggle datasets download -d chitwanmanchanda/fraudulent-transactions-data -p data")
    if rc != 0:
        print("⚠️ Kaggle CLI returned non-zero. Check installation/auth.")
        return
    for z in data_dir.glob("*.zip"):
        try:
            with zipfile.ZipFile(z, "r") as zip_ref:
                zip_ref.extractall(data_dir)
        finally:
            z.unlink(missing_ok=True)

if not csv_path.exists():
    print(f"CSV '{CSV_FILENAME}' not found in {DATA_DIR.resolve()}")
    if _kaggle_available():
        _download_from_kaggle(DATA_DIR)
    else:
        print("⚠️ Kaggle CLI not found. Place the dataset CSV into ./data")
    resolved = _find_csv_after_download(DATA_DIR)
    if resolved is None:
        raise FileNotFoundError("No CSV found in ./data after download/copy.")
    CSV_FILENAME = resolved.name
    csv_path = resolved
    print(f"✅ Using dataset file: {csv_path.name}")
else:
    print("✅ CSV already exists:", csv_path.name)

df = read_data_pandas(csv_path, quick_experiment=QUICK_EXPERIMENT, max_rows=MAX_ROWS)

print(f"\nDataset Columns: {df.columns.tolist()}")
print("\nFirst 3 rows:")
display(df.head(3))

# ============================================================================
# SECTION 3: EXPLORATORY DATA ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("EXPLORATORY DATA ANALYSIS")
print("="*80)

# Class Distribution
counts = df['isFraud'].value_counts().sort_index()
percentages = (counts / counts.sum()) * 100

print("\nClass Distribution:")
print(f"  Legitimate (0): {counts.get(0,0):,} ({percentages.get(0,0):.2f}%)")
print(f"  Fraud (1): {counts.get(1,0):,} ({percentages.get(1,0):.4f}%)")

# FIG 1: Class distribution
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(['Legitimate (0)', 'Fraud (1)'], [counts.get(0,0), counts.get(1,0)],
              color=['#2ecc71', '#e74c3c'])
ax.set_title("Class Distribution (isFraud)", fontweight='bold')
ax.set_ylabel("Number of Transactions")
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x):,}"))
for bar, count, pct in zip(bars, [counts.get(0,0), counts.get(1,0)], 
                           [percentages.get(0,0), percentages.get(1,0)]):
    x = bar.get_x() + bar.get_width() / 2
    ax.text(x, count * 0.5 if count>0 else 0.1, f"{int(count):,}\n({pct:.2f}%)",
            ha='center', va='center', fontsize=10, color='white', fontweight='bold')
plt.tight_layout()
plt.savefig('figures/01_class_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Transaction Type Distribution + Fraud by Type
counts_type = df["type"].value_counts()
percentages_type = (counts_type / counts_type.sum()) * 100

print("\nTransaction Type Distribution:")
for typ, cnt, pct in zip(counts_type.index, counts_type.values, percentages_type.values):
    print(f"  {typ}: {cnt:,} ({pct:.2f}%)")

fraud_by_type = df.groupby('type')['isFraud'].agg(['sum', 'count', 'mean']).reset_index()
fraud_by_type.columns = ['type', 'Fraud_Count', 'Total_Count', 'Fraud_Rate']
fraud_by_type['Fraud_Rate'] = fraud_by_type['Fraud_Rate'] * 100

print("\nFraud by Transaction Type:")
print(fraud_by_type)

# FIG 2: Fraud by type (counts)
plt.figure(figsize=(8,5))
order = fraud_by_type.sort_values('Fraud_Count', ascending=False)
plt.bar(order['type'], order['Fraud_Count'])
plt.xlabel("Transaction Type")
plt.ylabel("Fraud Count")
plt.title("Fraud Count by Transaction Type", fontweight='bold')
plt.tight_layout()
plt.savefig('figures/02_fraud_by_type.png', dpi=300, bbox_inches='tight')
plt.close()

# EDA numbers needed for report
median_fraud_amount = float(df.loc[df['isFraud']==1, 'amount'].median())
median_legit_amount = float(df.loc[df['isFraud']==0, 'amount'].median())

# For narrative stats:
# - % of fraudulent CASH_OUT with origin drained (newbalanceOrig == 0)
fraud_cashout = df[(df['isFraud']==1) & (df['type']=='CASH_OUT')]
pct_cashout_drained = 100.0 * (fraud_cashout['newbalanceOrig']==0).mean() if len(fraud_cashout) else np.nan

# - % of fraudulent TRANSFER with destination unchanged (oldbalanceDest == newbalanceDest and amount>0)
fraud_transfer = df[(df['isFraud']==1) & (df['type']=='TRANSFER')]
pct_transfer_dest_unchanged = 100.0 * (((fraud_transfer['oldbalanceDest']==fraud_transfer['newbalanceDest']) & 
                                        (fraud_transfer['amount']>0)).mean()) if len(fraud_transfer) else np.nan

# ============================================================================
# SECTION 4: SCHEMA VALIDATION
# ============================================================================

print("\n" + "="*80)
print("SCHEMA VALIDATION")
print("="*80)

EXPECTED_COLS = {
    "step","type","amount","nameOrig","oldbalanceOrg","newbalanceOrig",
    "nameDest","oldbalanceDest","newbalanceDest","isFraud","isFlaggedFraud"
}

DTYPE_MAP = {
    "step": "int32",
    "type": "category",
    "amount": "float32",
    "nameOrig": "category",
    "oldbalanceOrg": "float32",
    "newbalanceOrig": "float32",
    "nameDest": "category",
    "oldbalanceDest": "float32",
    "newbalanceDest": "float32",
    "isFraud": "int8",
    "isFlaggedFraud": "int8",
}

def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and enforce schema."""
    df = df.copy()
    
    missing = EXPECTED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"❌ Missing columns: {sorted(missing)}")
    
    for c, t in DTYPE_MAP.items():
        if t == "category":
            df[c] = df[c].astype("category")
        else:
            df[c] = pd.to_numeric(df[c], errors="raise").astype(t)
    
    # Validate binary targets
    for bcol in ("isFraud", "isFlaggedFraud"):
        uniq = set(pd.Series(df[bcol]).dropna().unique().tolist())
        if not uniq.issubset({0, 1}):
            raise ValueError(f"Column '{bcol}' must be binary 0/1")
    
    print("✅ Schema & dtypes validated")
    return df

df = enforce_schema(df)

# ============================================================================
# SECTION 5: FEATURE ENGINEERING
# ============================================================================

print("\n" + "="*80)
print("FEATURE ENGINEERING")
print("="*80)

def add_engineered_features(df: pd.DataFrame) -> tuple:
    """
    Add comprehensive engineered features for fraud detection.
    Returns: (df_out, numeric_feats, categorical_feats)
    """
    df = df.copy()
    
    # Cast to float32 for consistency
    for c in ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]:
        df[c] = df[c].astype("float32")
    
    # Balance deltas (discrepancy detection)
    df["tx_delta_orig"] = (df["oldbalanceOrg"] - df["newbalanceOrig"] - df["amount"]).astype("float32")
    df["tx_delta_dest"] = (df["newbalanceDest"] - df["oldbalanceDest"]).astype("float32")
    
    # Ratios (safe divide)
    denom_orig = df["oldbalanceOrg"]
    denom_dest = df["oldbalanceDest"]
    num = df["amount"]
    
    df["orig_balance_ratio"] = np.divide(
        num, denom_orig, out=np.zeros_like(num, dtype="float32"), where=(denom_orig > 0)
    ).astype("float32")
    
    df["dest_balance_ratio"] = np.divide(
        num, denom_dest, out=np.zeros_like(num, dtype="float32"), where=(denom_dest > 0)
    ).astype("float32")
    
    # Binary flags
    df["flag_orig_negative"] = (df["newbalanceOrig"] < 0).astype("int8")
    df["flag_dest_negative"] = (df["newbalanceDest"] < 0).astype("int8")
    df["flag_orig_nochange"] = ((df["oldbalanceOrg"] == df["newbalanceOrig"]) & (df["amount"] > 0)).astype("int8")
    df["flag_dest_nochange"] = ((df["oldbalanceDest"] == df["newbalanceDest"]) & (df["amount"] > 0)).astype("int8")
    df["flag_high_risk_type"] = df["type"].isin(["CASH_OUT", "TRANSFER"]).astype("int8")
    
    # Log transform
    df["log_amount"] = np.log1p(df["amount"]).astype("float32")
    
    # Zero balance indicators
    df["origZeroBalanceAfter"] = (df["newbalanceOrig"] == 0).astype("int8")
    df["destZeroBalanceBefore"] = (df["oldbalanceDest"] == 0).astype("int8")
    
    # Temporal features
    df["hourOfDay"] = (df["step"] % 24).astype("int8")
    df["dayOfMonth"] = (df["step"] // 24).astype("int16")
    
    # Drop high-cardinality IDs
    df.drop(columns=["nameOrig", "nameDest"], inplace=True, errors="ignore")
    
    # Sanitize infinities and NaNs
    num_cols = df.select_dtypes(include=["float32", "float64"]).columns
    if not np.isfinite(df[num_cols].to_numpy()).all():
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
    if df[num_cols].isna().any().any():
        df[num_cols] = df[num_cols].fillna(0.0)
    
    # Feature lists
    numeric_feats = [
        "step", "amount",
        "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest",
        "tx_delta_orig", "tx_delta_dest",
        "orig_balance_ratio", "dest_balance_ratio",
        "flag_orig_negative", "flag_dest_negative",
        "flag_orig_nochange", "flag_dest_nochange",
        "flag_high_risk_type",
        "log_amount",
        "origZeroBalanceAfter", "destZeroBalanceBefore",
        "hourOfDay", "dayOfMonth"
    ]
    categorical_feats = ["type"]
    
    df["type"] = df["type"].astype("string")
    return df, numeric_feats, categorical_feats

df, numeric_feats, categorical_feats = add_engineered_features(df)

print("✅ Feature engineering complete")
print(f"   Total features: {df.shape[1]}")
print(f"   Numeric features: {len(numeric_feats)}")
print(f"   Categorical features: {len(categorical_feats)}")

# ============================================================================
# SECTION 6: TIME-AWARE TRAIN/TEST SPLIT (robust)
# ============================================================================

def downsample_majority(df: pd.DataFrame, target: str = "isFraud", ratio: int = DS_RATIO, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Keep all fraud; sample 'ratio' legitimate per fraud in TRAIN only."""
    fraud = df[df[target] == 1]
    legit = df[df[target] == 0]
    if fraud.empty:
        return df
    n_keep = min(len(fraud) * ratio, len(legit))
    legit_s = legit.sample(n=n_keep, random_state=seed)
    return (pd.concat([fraud, legit_s])
              .sample(frac=1.0, random_state=seed)
              .reset_index(drop=True))

print("\n" + "="*80)
print("TIME-AWARE TRAIN/TEST SPLIT")
print("="*80)

def time_aware_split(
    df_in: pd.DataFrame,
    cutoff: int,
    target_col: str = "isFraud",
    verbose: bool = True
) -> tuple:
    """Chronological split; robust to empty splits (falls back to 80/20 by step)."""
    work = df_in.copy()
    work["step"] = pd.to_numeric(work["step"], errors="coerce")
    work = work.dropna(subset=["step"])

    def _do_split(w, c):
        tr = w.loc[w["step"] <= c].copy().sort_values("step").reset_index(drop=True)
        te = w.loc[w["step"] >  c].copy().sort_values("step").reset_index(drop=True)
        return tr, te

    df_train, df_test = _do_split(work, cutoff)
    if df_train.empty or df_test.empty:
        smin, smax = int(work["step"].min()), int(work["step"].max())
        fallback_cutoff = int(np.floor(work["step"].quantile(0.8)))
        print(f"⚠️ Provided cutoff {cutoff} yielded an empty split for steps in [{smin}, {smax}]. "
              f"Falling back to 80% quantile cutoff = {fallback_cutoff}.")
        cutoff = fallback_cutoff
        df_train, df_test = _do_split(work, cutoff)

    if verbose:
        def _safe_minmax(series):
            if series.empty:
                return "—", "—"
            return int(series.min()), int(series.max())

        tr_min, tr_max = _safe_minmax(df_train["step"])
        te_min, te_max = _safe_minmax(df_test["step"])
        tr_pos, te_pos = int(df_train[target_col].sum()), int(df_test[target_col].sum())
        tr_rate, te_rate = df_train[target_col].mean(), df_test[target_col].mean()

        print(f"✅ Train: {df_train.shape} | steps [{tr_min}, {tr_max}]")
        print(f"   Fraud cases: {tr_pos:,} ({tr_rate:.4%})")
        print(f"✅ Test:  {df_test.shape} | steps [{te_min}, {te_max}]")
        print(f"   Fraud cases: {te_pos:,} ({te_rate:.4%})")

    return df_train, df_test, cutoff

df_train, df_test, USED_CUTOFF = time_aware_split(df, cutoff=TRAIN_CUTOFF_STEP)

# Prepare features for train/test
feature_cols = numeric_feats + categorical_feats

# Downsample TRAIN during dev for speed
if FAST_DEV:
    df_train_fit = downsample_majority(df_train, ratio=DS_RATIO)
    print(df_train_fit["isFraud"].value_counts(normalize=True).rename("train class ratio"))
else:
    df_train_fit = df_train

X_train = df_train_fit[feature_cols].copy()
y_train = df_train_fit["isFraud"].astype("int8").values
X_test = df_test[feature_cols].copy()
y_test = df_test["isFraud"].astype("int8").values

# Ensure categoricals are strings
for c in categorical_feats:
    X_train[c] = X_train[c].astype("string")
    X_test[c]  = X_test[c].astype("string")

print("\n✅ Feature/Target separation complete")
print(f"   X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"   X_test: {X_test.shape}, y_test: {y_test.shape}")

# ============================================================================
# SECTION 7: MODELS + THRESHOLD TUNING
# ============================================================================

def create_preprocess(numeric_feats, categorical_feats):
    """Sparse-friendly preproc: MaxAbsScaler + OHE (sparse)."""
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=True, dtype=np.float32)

    preprocess = ColumnTransformer(
        transformers=[
            ("num", MaxAbsScaler(), numeric_feats),
            ("cat", ohe, categorical_feats),
        ],
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=False
    )
    return preprocess

preprocess = create_preprocess(numeric_feats, categorical_feats)

LR_CLASS_WEIGHT = None if FAST_DEV else "balanced"
RF_CLASS_WEIGHT = None if FAST_DEV else "balanced"

lr_pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("clf", LogisticRegression(
        solver="sag",
        class_weight=LR_CLASS_WEIGHT,
        max_iter=LR_MAX_ITER_DEV if FAST_DEV else 1000,
        tol=LR_TOL_DEV if FAST_DEV else 1e-4,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1
    ))
], memory=memory)

rf_pipe = Pipeline(steps=[
    ("preprocess", clone(preprocess)),
    ("clf", RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=50,
        min_samples_leaf=20,
        max_features="sqrt",
        class_weight=RF_CLASS_WEIGHT,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    ))
], memory=None)

def threshold_table(
    y_true, y_scores,
    sort_by="f1",
    descending=True,
    beta=1.0,
    min_precision=None,
    min_recall=None,
    grid_points=201
):
    """Build threshold sweep table."""
    EPS = 1e-12
    y_true = np.asarray(y_true).astype(int).ravel()
    y_scores = np.asarray(y_scores, dtype=float).ravel()
    qs = np.linspace(0.0, 1.0, grid_points)
    thresholds = np.quantile(y_scores, qs)
    thresholds = np.unique(np.clip(thresholds, 0.0, 1.0))
    rows = []
    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        precision = tp / (tp + fp + EPS)
        recall = tp / (tp + fn + EPS)
        f1 = 2 * precision * recall / (precision + recall + EPS)
        b2 = beta * beta
        fbeta = (1 + b2) * (precision * recall) / (b2 * precision + recall + EPS)
        rows.append({
            "threshold": float(t),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fbeta": fbeta,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)
        })
    df_tbl = pd.DataFrame(rows)
    mask = pd.Series(True, index=df_tbl.index)
    if min_precision is not None:
        mask &= df_tbl["precision"] >= float(min_precision)
    if min_recall is not None:
        mask &= df_tbl["recall"] >= float(min_recall)
    filtered = df_tbl.loc[mask].copy()
    if filtered.empty:
        # Fallback: best F1 if constraint infeasible
        filtered = df_tbl.copy()
        sort_by = "f1"
        print("⚠️ No threshold met the constraint; falling back to best F1.")
    filtered = filtered.sort_values(sort_by, ascending=not descending).reset_index(drop=True)
    best_row = filtered.iloc[0].copy()
    return filtered, best_row

# ---- Train + time Logistic Regression
print("\nTraining Logistic Regression...")
t0 = time.time()
lr_pipe.fit(X_train, y_train)
lr_train_time = time.time() - t0
print(f"✅ LR training done in {lr_train_time:.2f} sec")

# Predict + time
t0 = time.time()
proba_lr = lr_pipe.predict_proba(X_test)[:, 1]
lr_pred_time = time.time() - t0

roc_auc_lr = roc_auc_score(y_test, proba_lr)
pr_auc_lr = average_precision_score(y_test, proba_lr)

tbl_lr, best_lr = threshold_table(
    y_test, proba_lr,
    sort_by="precision",
    min_recall=0.95,
    beta=2.0
)
CHOSEN_THRESHOLD_LR = float(best_lr["threshold"])
pred_lr = (proba_lr >= CHOSEN_THRESHOLD_LR).astype(int)

tn_lr, fp_lr, fn_lr, tp_lr = confusion_matrix(y_test, pred_lr).ravel()
precision_lr = precision_score(y_test, pred_lr, zero_division=0)
recall_lr = recall_score(y_test, pred_lr, zero_division=0)
f1_lr = f1_score(y_test, pred_lr, zero_division=0)

# ---- Train + time Random Forest
print("\nTraining Random Forest...")
t0 = time.time()
rf_pipe.fit(X_train, y_train)
rf_train_time = time.time() - t0
print(f"✅ RF training done in {rf_train_time:.2f} sec")

# Predict + time
t0 = time.time()
proba_rf = rf_pipe.predict_proba(X_test)[:, 1]
rf_pred_time = time.time() - t0

tbl_rf, best_rf = threshold_table(
    y_test, proba_rf,
    sort_by="precision",
    min_recall=0.95,
    beta=2.0
)
CHOSEN_THRESHOLD_RF = float(best_rf["threshold"])
pred_rf = (proba_rf >= CHOSEN_THRESHOLD_RF).astype(int)

tn_rf, fp_rf, fn_rf, tp_rf = confusion_matrix(y_test, pred_rf).ravel()
precision_rf = precision_score(y_test, pred_rf, zero_division=0)
recall_rf = recall_score(y_test, pred_rf, zero_division=0)
f1_rf = f1_score(y_test, pred_rf, zero_division=0)
roc_auc_rf = roc_auc_score(y_test, proba_rf)
pr_auc_rf = average_precision_score(y_test, proba_rf)

# ============================================================================
# SECTION 8: FIGURES FOR THE REPORT (10 total)
# ============================================================================

# FIG 3: LR confusion matrix
plt.figure(figsize=(6,5))
ConfusionMatrixDisplay.from_predictions(
    y_test, pred_lr,
    display_labels=["Legitimate", "Fraud"],
    values_format=",.0f", colorbar=False, cmap='Blues'
)
plt.title(f'Logistic Regression Confusion Matrix (τ={CHOSEN_THRESHOLD_LR:.3f})', fontweight='bold')
plt.tight_layout()
plt.savefig('figures/03_lr_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# FIG 4: RF confusion matrix
plt.figure(figsize=(6,5))
ConfusionMatrixDisplay.from_predictions(
    y_test, pred_rf,
    display_labels=["Legitimate", "Fraud"],
    values_format=",.0f", colorbar=False, cmap='Greens'
)
plt.title(f'Random Forest Confusion Matrix (τ={CHOSEN_THRESHOLD_RF:.3f})', fontweight='bold')
plt.tight_layout()
plt.savefig('figures/04_rf_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# Feature Importance (and figure 5)
feature_names_out = rf_pipe.named_steps['preprocess'].get_feature_names_out()
feature_importance = pd.DataFrame({
    'feature': feature_names_out,
    'importance': rf_pipe.named_steps['clf'].feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Importance Score', fontsize=12)
plt.title('Top 15 Feature Importances - Random Forest', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('figures/05_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# FIG 6: Side-by-side confusion matrices
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, pred_lr,
    display_labels=["Legitimate", "Fraud"],
    ax=ax1, values_format=",.0f", colorbar=False, cmap='Blues'
)
ax1.set_title(f'Logistic Regression (τ={CHOSEN_THRESHOLD_LR:.3f})', fontweight='bold')
ConfusionMatrixDisplay.from_predictions(
    y_test, pred_rf,
    display_labels=["Legitimate", "Fraud"],
    ax=ax2, values_format=",.0f", colorbar=False, cmap='Greens'
)
ax2.set_title(f'Random Forest (τ={CHOSEN_THRESHOLD_RF:.3f})', fontweight='bold')
plt.tight_layout()
plt.savefig('figures/06_confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# FIG 7: ROC curves comparison
plt.figure(figsize=(8,6))
fpr_lr, tpr_lr, _ = roc_curve(y_test, proba_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, proba_rf)
plt.plot(fpr_lr, tpr_lr, label=f"LR (ROC-AUC={roc_auc_lr:.4f})")
plt.plot(fpr_rf, tpr_rf, label=f"RF (ROC-AUC={roc_auc_rf:.4f})")
plt.plot([0,1],[0,1],'k--', lw=1)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curves: LR vs RF", fontweight='bold'); plt.legend()
plt.tight_layout()
plt.savefig('figures/07_roc_curves_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# FIG 8: Precision-Recall curves comparison
plt.figure(figsize=(8,6))
prec_lr, rec_lr, _ = precision_recall_curve(y_test, proba_lr)
prec_rf, rec_rf, _ = precision_recall_curve(y_test, proba_rf)
plt.plot(rec_lr, prec_lr, label=f"LR (PR-AUC={pr_auc_lr:.4f})")
plt.plot(rec_rf, prec_rf, label=f"RF (PR-AUC={pr_auc_rf:.4f})")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision-Recall Curves: LR vs RF", fontweight='bold'); plt.legend()
plt.tight_layout()
plt.savefig('figures/08_precision_recall_curves_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# FIG 9: Metrics comparison bar chart
metrics_names = ["Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]
lr_vals = [precision_lr, recall_lr, f1_lr, roc_auc_lr, pr_auc_lr]
rf_vals = [precision_rf, recall_rf, f1_rf, roc_auc_rf, pr_auc_rf]

x = np.arange(len(metrics_names)); width = 0.35
plt.figure(figsize=(10,6))
plt.bar(x - width/2, lr_vals, width, label="LR")
plt.bar(x + width/2, rf_vals, width, label="RF")
plt.xticks(x, metrics_names)
plt.ylabel("Score"); plt.ylim(0, 1.05)
plt.title("Metrics Comparison: LR vs RF", fontweight='bold'); plt.legend()
plt.tight_layout()
plt.savefig('figures/09_metrics_comparison_bar.png', dpi=300, bbox_inches='tight')
plt.close()

# FIG 10: FP & FN comparison
labels = ["FP", "FN"]
lr_counts = [fp_lr, fn_lr]
rf_counts = [fp_rf, fn_rf]
x = np.arange(len(labels)); width = 0.35
plt.figure(figsize=(8,6))
plt.bar(x - width/2, lr_counts, width, label="LR")
plt.bar(x + width/2, rf_counts, width, label="RF")
plt.xticks(x, labels)
plt.ylabel("Count")
plt.title("False Positives & False Negatives: LR vs RF", fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('figures/10_fp_fn_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# SECTION 9: COMPARISON TABLE + RESULTS EXPORT
# ============================================================================

comparison_df = pd.DataFrame([
    {
        "Model": "Logistic Regression",
        "Threshold": CHOSEN_THRESHOLD_LR,
        "Precision": precision_lr,
        "Recall": recall_lr,
        "F1-Score": f1_lr,
        "ROC-AUC": roc_auc_lr,
        "PR-AUC": pr_auc_lr,
        "TP": tp_lr, "FP": fp_lr, "FN": fn_lr, "TN": tn_lr
    },
    {
        "Model": "Random Forest",
        "Threshold": CHOSEN_THRESHOLD_RF,
        "Precision": precision_rf,
        "Recall": recall_rf,
        "F1-Score": f1_rf,
        "ROC-AUC": roc_auc_rf,
        "PR-AUC": pr_auc_rf,
        "TP": tp_rf, "FP": fp_rf, "FN": fn_rf, "TN": tn_rf
    }
])

print("\nModel Performance Comparison:")
display(comparison_df[[
    "Model","Threshold","Precision","Recall","F1-Score","ROC-AUC","PR-AUC","TP","FP","FN","TN"
]])

comparison_df.to_csv('model_comparison.csv', index=False)
print("\n✅ Comparison table saved to 'model_comparison.csv'")

# Derived stats for report text
fpr_lr_pct = 100.0 * fp_lr / (fp_lr + tn_lr) if (fp_lr + tn_lr) else np.nan
alerts_lr = tp_lr + fp_lr
test_size = int(len(y_test))
alerts_lr_pct = 100.0 * alerts_lr / test_size if test_size else np.nan

fpr_rf_pct = 100.0 * fp_rf / (fp_rf + tn_rf) if (fp_rf + tn_rf) else np.nan
alerts_rf = tp_rf + fp_rf
alerts_rf_pct = 100.0 * alerts_rf / test_size if test_size else np.nan

# Training/prediction time (minutes + speedups)
lr_train_min = lr_train_time / 60.0
rf_train_min = rf_train_time / 60.0
lr_pred_ms = lr_pred_time * 1000.0 / max(test_size,1)
rf_pred_ms = rf_pred_time * 1000.0 / max(test_size,1)

train_speedup = (rf_train_time / lr_train_time) if lr_train_time>0 else np.nan
pred_speedup = (rf_pred_time / lr_pred_time) if lr_pred_time>0 else np.nan

# Top-5 features (RF)
top5 = feature_importance.head(5).reset_index(drop=True)

# RESULTS TXT (for easy copy into report)
with open("results.txt", "w", encoding="utf-8") as f:
    f.write("==== Exploratory Data Analysis ====\n")
    f.write(f"Median fraud amount: {median_fraud_amount:,.0f} units\n")
    f.write(f"Median legitimate amount: {median_legit_amount:,.0f} units\n")
    f.write(f"% fraudulent CASH_OUT with origin drained (newbalanceOrig==0): {pct_cashout_drained:.2f}%\n")
    f.write(f"% fraudulent TRANSFER with dest unchanged: {pct_transfer_dest_unchanged:.2f}%\n")
    f.write("\n")

    f.write("==== Logistic Regression (Results at Optimal Threshold) ====\n")
    f.write(f"Threshold: {CHOSEN_THRESHOLD_LR:.4f}\n")
    f.write(f"Precision: {precision_lr:.4f}\n")
    f.write(f"Recall: {recall_lr:.4f}\n")
    f.write(f"F1-Score: {f1_lr:.4f}\n")
    f.write(f"ROC-AUC: {roc_auc_lr:.4f}\n")
    f.write(f"PR-AUC: {pr_auc_lr:.4f}\n")
    f.write(f"TP: {tp_lr:,}\nFP: {fp_lr:,}\nFN: {fn_lr:,}\nTN: {tn_lr:,}\n")
    f.write(f"False Positive Rate: {fp_lr:,} / {fp_lr + tn_lr:,} = {fpr_lr_pct:.4f}%\n")
    f.write(f"Alerts generated: {alerts_lr:,} of {test_size:,} = {alerts_lr_pct:.2f}%\n")
    f.write(f"Training time: {lr_train_min:.2f} min; Prediction: {lr_pred_ms:.4f} ms/example\n")
    f.write("\n")

    f.write("==== Random Forest (Results at Optimal Threshold) ====\n")
    f.write(f"Threshold: {CHOSEN_THRESHOLD_RF:.4f}\n")
    f.write(f"Precision: {precision_rf:.4f}\n")
    f.write(f"Recall: {recall_rf:.4f}\n")
    f.write(f"F1-Score: {f1_rf:.4f}\n")
    f.write(f"ROC-AUC: {roc_auc_rf:.4f}\n")
    f.write(f"PR-AUC: {pr_auc_rf:.4f}\n")
    f.write(f"TP: {tp_rf:,}\nFP: {fp_rf:,}\nFN: {fn_rf:,}\nTN: {tn_rf:,}\n")
    f.write(f"False Positive Rate: {fp_rf:,} / {fp_rf + tn_rf:,} = {fpr_rf_pct:.4f}%\n")
    f.write(f"Alerts generated: {alerts_rf:,} of {test_size:,} = {alerts_rf_pct:.2f}%\n")
    f.write(f"Training time: {rf_train_min:.2f} min; Prediction: {rf_pred_ms:.4f} ms/example\n")
    f.write("\n")

    f.write("==== Comparative Analysis ====\n")
    f.write(f"Precision @ Recall≥95% — LR: {precision_lr:.4f}, RF: {precision_rf:.4f}\n")
    f.write(f"Recall — LR: {recall_lr:.4f}, RF: {recall_rf:.4f}\n")
    f.write(f"F1-Score — LR: {f1_lr:.4f}, RF: {f1_rf:.4f}\n")
    f.write(f"ROC-AUC — LR: {roc_auc_lr:.4f}, RF: {roc_auc_rf:.4f}\n")
    f.write(f"PR-AUC — LR: {pr_auc_lr:.4f}, RF: {pr_auc_rf:.4f}\n")
    if not np.isnan(train_speedup):
        faster = "LR" if train_speedup>1 else "RF"
        factor = train_speedup if train_speedup>1 else (1/train_speedup if train_speedup>0 else np.nan)
        f.write(f"Training time ratio (RF/LR): {train_speedup:.2f} → {('LR' if faster=='LR' else 'RF')} ~{factor:.1f}× faster\n")
    if not np.isnan(pred_speedup):
        faster_p = "LR" if pred_speedup>1 else "RF"
        factor_p = pred_speedup if pred_speedup>1 else (1/pred_speedup if pred_speedup>0 else np.nan)
        f.write(f"Prediction time ratio (RF/LR): {pred_speedup:.2f} → {('LR' if faster_p=='LR' else 'RF')} ~{factor_p:.1f}× faster per example\n")
    f.write("\n")

    f.write("Top 5 Features by RF importance:\n")
    for i, row in top5.iterrows():
        f.write(f"{i+1}. {row['feature']} ({100*row['importance']:.2f}% importance)\n")

print("✅ Wrote 'results.txt' with all values for your report.")
print("✅ Saved all 10 figures under './figures' with the expected names.")
