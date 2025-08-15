# =========================
# XGBoost binary classifier (leakage-safe, CV-tuned, threshold-tuned)
# =========================
# Requirements:
#   pip install xgboost imbalanced-learn scikit-learn pandas numpy matplotlib
# -------------------------

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GroupShuffleSplit, GroupKFold, RandomizedSearchCV, cross_val_predict
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, f1_score
)

from xgboost import XGBClassifier
from scipy import stats


# -------------------------
# 0) CONFIG
# -------------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Set your binary target column here (0/1):
TARGET_COL = "numercnt"

# Columns that should NEVER be features (drop if they exist)
ID_COLS = [
    "measurement_year", "asdb_plankey", "member_key", "src_mbr_id",
    "indiv_anlytcs_id", "cust_no", "cust_nm", "clssdt"
]

# Treat these as categorical even if they are integers in the CSV
FORCE_CATEGORICAL = [
    "zip_code", "state", "lob_cd", "submeasurekey", "racetype", "ethnicitytype", "city"
]

# -------------------------
# 1) LOAD YOUR DATA
# -------------------------
# Expecting a DataFrame named `df` in memory OR load it here.
# Example:
# df = pd.read_csv("your_data.csv")

assert TARGET_COL in df.columns, f"Target column '{TARGET_COL}' not found in df."

# Optional: ensure binary target is 0/1 integers
df = df.dropna(subset=[TARGET_COL]).copy()
df[TARGET_COL] = (df[TARGET_COL].astype(float) > 0).astype(int)


# -------------------------
# 2) TRAIN/TEST SPLIT (before ANY fitting to avoid leakage)
#    If member_key exists, we do a group-aware split so people
#    don't leak across train/test.
# -------------------------
groups = df["member_key"] if "member_key" in df.columns else None

feature_cols = [c for c in df.columns if c not in set(ID_COLS + [TARGET_COL])]
X_full = df[feature_cols].copy()
y_full = df[TARGET_COL].values

if groups is not None:
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X_full, y_full, groups))
    X_train, X_test = X_full.iloc[train_idx], X_full.iloc[test_idx]
    y_train, y_test = y_full[train_idx], y_full[test_idx]
    groups_train = groups.iloc[train_idx]
    groups_test = groups.iloc[test_idx]
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.2, stratify=y_full, random_state=RANDOM_STATE
    )
    groups_train = None


# -------------------------
# 3) FEATURE TYPE INFERENCE
# -------------------------
# Force some columns to categorical even if they look numeric
force_cats_present = [c for c in FORCE_CATEGORICAL if c in X_train.columns]

cat_cols = list(
    set(list(X_train.select_dtypes(include=["object", "category", "bool"]).columns) + force_cats_present)
)
num_cols = [c for c in X_train.columns if c not in cat_cols]

# Build preprocessing for numeric and categorical columns
num_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Use `sparse=False` for broad sklearn compatibility
cat_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ],
    remainder="drop"
)


# -------------------------
# 4) MODEL & PIPELINE
#    Oversampling happens ONLY in training folds via the pipeline.
# -------------------------
xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",        # fast/robust default
    random_state=RANDOM_STATE,
    n_jobs=-1
)

pipeline = ImbPipeline(steps=[
    ("pre", preprocessor),
    ("ros", RandomOverSampler(random_state=RANDOM_STATE)),
    ("clf", xgb)
])

# -------------------------
# 5) HYPERPARAMETER SEARCH (Randomized)
#    We score on F1 to align with your business metric.
# -------------------------
param_distributions = {
    "clf__n_estimators": stats.randint(300, 1000),
    "clf__max_depth": stats.randint(3, 9),
    "clf__learning_rate": stats.loguniform(1e-2, 2e-1),
    "clf__subsample": stats.uniform(0.6, 0.4),        # 0.6..1.0
    "clf__colsample_bytree": stats.uniform(0.6, 0.4), # 0.6..1.0
    "clf__min_child_weight": stats.randint(1, 6),
    "clf__reg_alpha": stats.loguniform(1e-4, 1.0),
    "clf__reg_lambda": stats.loguniform(1e-2, 3.0),
}

if groups_train is not None:
    cv = GroupKFold(n_splits=5)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=40,
        scoring="f1",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
        verbose=1
    )
    search.fit(X_train, y_train, groups=groups_train)
else:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=40,
        scoring="f1",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
        verbose=1
    )
    search.fit(X_train, y_train)

best_pipe = search.best_estimator_


# -------------------------
# 6) DECISION THRESHOLD TUNING ON TRAINING (via CV OOF preds)
# -------------------------
# We choose the probability cutoff that maximizes F1 without touching the test set.
if groups_train is not None:
    oof_proba = cross_val_predict(
        best_pipe, X_train, y_train,
        cv=cv, method="predict_proba", n_jobs=-1, groups=groups_train
    )[:, 1]
else:
    oof_proba = cross_val_predict(
        best_pipe, X_train, y_train,
        cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]

thresholds = np.linspace(0.05, 0.95, 181)  # fine grid
f1s = [f1_score(y_train, (oof_proba >= t).astype(int)) for t in thresholds]
best_threshold = float(thresholds[int(np.argmax(f1s))])

print(f"\nBest CV F1 threshold on training = {best_threshold:.3f}  (F1={max(f1s):.4f})")

# Refit on full training set (the search already refit, but we do it once more explicitly)
if groups_train is not None:
    best_pipe.fit(X_train, y_train, ros__sampling_strategy="auto")
else:
    best_pipe.fit(X_train, y_train)


# -------------------------
# 7) EVALUATE ON THE HELD-OUT TEST SET (untouched)
# -------------------------
proba_test = best_pipe.predict_proba(X_test)[:, 1]
y_pred_test = (proba_test >= best_threshold).astype(int)

# Metrics
roc_auc = roc_auc_score(y_test, proba_test)
precision, recall, pr_thresh = precision_recall_curve(y_test, proba_test)
pr_auc = average_precision_score(y_test, proba_test)
f1 = f1_score(y_test, y_pred_test)

print("\nClassification Report (Test):")
print(classification_report(y_test, y_pred_test, digits=3))
print(f"ROC-AUC (Test): {roc_auc:.4f}")
print(f"PR-AUC  (Test): {pr_auc:.4f}")
print(f"F1      (Test): {f1:.4f}")

cm = confusion_matrix(y_test, y_pred_test)

# -------------------------
# 8) PLOTS (saved to files)
# -------------------------
plt.figure()
fpr, tpr, _ = roc_curve(y_test, proba_test)
plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Test)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve_test.png", dpi=200)
plt.close()

plt.figure()
plt.plot(recall, precision, label=f"PR (AP={pr_auc:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (Test)")
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig("pr_curve_test.png", dpi=200)
plt.close()

plt.figure()
# Confusion matrix plot with counts
ax = plt.gca()
im = ax.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix (Test)")
plt.colorbar(im, fraction=0.046, pad=0.04)
tick_marks = np.arange(2)
plt.xticks(tick_marks, [0, 1])
plt.yticks(tick_marks, [0, 1])
plt.xlabel("Predicted")
plt.ylabel("True")
# Add counts
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], "d"),
                ha="center", va="center")
plt.tight_layout()
plt.savefig("confusion_matrix_test.png", dpi=200)
plt.close()

print("\nSaved plots:")
print(" - roc_curve_test.png")
print(" - pr_curve_test.png")
print(" - confusion_matrix_test.png")

# -------------------------
# 9) WHAT TO REPORT
# -------------------------
#  - search.best_params_      -> tuned XGBoost hyperparameters
#  - best_threshold           -> cut-off used on test
#  - classification report, ROC-AUC, PR-AUC, F1 on test
# -------------------------
print("\nBest Hyperparameters:")
print(search.best_params_)
print(f"\nFinal decision threshold used on test: {best_threshold:.3f}")
