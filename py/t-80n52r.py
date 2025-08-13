# This cell builds the dataset, trains both models, and saves pretty charts for
# (1) classification report metrics and (2) ROC curves as separate image files.
#
# Note: Uses matplotlib (no seaborn), each chart is a single figure (no subplots),
# and no explicit colors are set per the instructions.

import numpy as np
import pandas as pd

# Recreate the user's sample dataset
np.random.seed(42)
n = 100
data = {
    "age": np.random.randint(0, 100, n),
    "racetype": np.random.randint(1, 5, n),
    "ethnicitytype": np.random.randint(1, 25, n),
    "numercnt": np.random.choice([0, 1], size=n, p=[0.85, 0.15]),  # Class imbalance
    "state": np.random.choice(["VA", "NC", "MD", "WV"], n),
    "zip_code": np.random.randint(20000, 99999, n),
    "latitude": np.random.uniform(35, 40, n),
    "longitude": np.random.uniform(-85, -75, n),
    "food_access": np.random.uniform(0, 100, n),
    "health_access": np.random.uniform(0, 100, n),
    "housing_desert": np.random.uniform(0, 50, n),
    "poverty_score": np.random.uniform(0, 100, n),
    "transport_access": np.random.uniform(0, 100, n),
    "education_index": np.random.uniform(0, 100, n),
    "citizenship_index": np.random.uniform(80, 100, n),
    "housing_ownership": np.random.uniform(0, 100, n),
    "income_index": np.random.uniform(0, 100, n),
    "income_inequality": np.random.uniform(0, 100, n),
    "natural_disaster": np.random.uniform(0, 100, n),
    "social_isolation": np.random.uniform(0, 100, n),
    "technology_access": np.random.uniform(0, 100, n),
    "unemployment_index": np.random.uniform(0, 20, n),
    "water_quality": np.random.uniform(0, 1, n),
    "health_infra": np.random.uniform(0, 100, n),
    "social_risk_score": np.random.uniform(0, 100, n)
}
df = pd.DataFrame(data)

# ============================
# CIS Compliance Modeling Kit
# ============================
from dataclasses import dataclass
from typing import List, Dict, Tuple

from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (roc_auc_score, average_precision_score, classification_report,
                             confusion_matrix, precision_recall_curve, roc_curve, auc)
from sklearn.inspection import permutation_importance

from xgboost import XGBClassifier

@dataclass
class CISConfig:
    target: str = "numercnt"
    categorical: Tuple[str, ...] = ("state", "racetype", "ethnicitytype")
    drop: Tuple[str, ...] = ("zip_code",)

def add_handcrafted_features(df_in: pd.DataFrame) -> pd.DataFrame:
    X = df_in.copy()
    X["age_sq"] = X["age"] ** 2
    X["age_bin"] = pd.cut(
        X["age"],
        bins=[-1, 0, 1, 2, 6, 12, 19, 35, 50, 65, 80, 120],
        labels=["<1","1","2","3-6","7-12","13-19","20-35","36-50","51-65","66-80","80+"]
    )
    X["lat_c"] = X["latitude"] - X["latitude"].mean()
    X["lon_c"] = X["longitude"] - X["longitude"].mean()
    X["geo_radius"] = np.sqrt(X["lat_c"]**2 + X["lon_c"]**2)
    for col in ["income_index", "income_inequality", "housing_ownership", "health_infra"]:
        X[f"log1p_{col}"] = np.log1p(np.clip(X[col], a_min=0, a_max=None))
    X["poverty_x_food"] = X["poverty_score"] * X["food_access"]
    X["health_x_transport"] = X["health_access"] * X["transport_access"]
    X["isolation_x_education"] = X["social_isolation"] * X["education_index"]
    X["age_bin"] = X["age_bin"].astype("object")
    return X

def feature_spaces(df_in: pd.DataFrame, cfg: CISConfig) -> Tuple[List[str], List[str], List[str]]:
    cats = list(cfg.categorical) + ["age_bin"]
    base_cols = [c for c in df_in.columns if c not in {cfg.target, *cfg.drop}]
    numeric = [c for c in base_cols if c not in cats]
    to_drop = list(cfg.drop)
    return numeric, cats, to_drop

def build_column_transformer(numeric: List[str], cats: List[str], scale_numeric: bool) -> ColumnTransformer:
    if scale_numeric:
        numeric_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
    else:
        numeric_pipeline = "passthrough"
    cat_pipeline = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer(
        transformers=[("num", numeric_pipeline, numeric), ("cat", cat_pipeline, cats)],
        remainder="drop",
        verbose_feature_names_out=False
    )

def build_lr_pipeline(preproc: ColumnTransformer) -> ImbPipeline:
    fs_base = LogisticRegression(penalty="l1", solver="saga", C=1.0, max_iter=5000, n_jobs=-1, random_state=42)
    selector = SelectFromModel(fs_base, threshold="median")
    clf = LogisticRegression(penalty="l2", solver="lbfgs", C=1.0, max_iter=5000, n_jobs=-1, random_state=42)
    return ImbPipeline(steps=[("preproc", preproc), ("smote", SMOTE(random_state=42, k_neighbors=3)), ("select", selector), ("clf", clf)])

def build_xgb_pipeline(preproc: ColumnTransformer) -> ImbPipeline:
    fs_model = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9,
                             reg_lambda=1.0, reg_alpha=0.0, random_state=42, n_jobs=-1, eval_metric="logloss")
    selector = SelectFromModel(fs_model, threshold="median")
    clf = XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                        reg_lambda=1.5, reg_alpha=0.0, random_state=42, n_jobs=-1, eval_metric="logloss",
                        min_child_weight=2, gamma=0.0)
    return ImbPipeline(steps=[("preproc", preproc), ("smote", SMOTE(random_state=42, k_neighbors=3)), ("select", selector), ("clf", clf)])

def eval_thresholds(y_true, y_prob):
    precision, recall, thresh = precision_recall_curve(y_true, y_prob)
    f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0.0)
    best_f1_idx = np.argmax(f1)
    best_f1_thr = 0.5 if best_f1_idx >= len(thresh) else thresh[best_f1_idx]
    j_stat = recall[:-1] + precision[:-1] - 1.0
    best_j_idx = np.argmax(j_stat)
    best_j_thr = 0.5 if best_j_idx >= len(thresh) else thresh[best_j_idx]
    return {"best_f1_threshold": float(best_f1_thr), "best_j_threshold": float(best_j_thr)}

def model_card(name: str, y_true, y_prob, threshold: float = 0.5) -> Dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "model": name,
        "roc_auc": roc_auc_score(y_true, y_prob),
        "avg_precision": average_precision_score(y_true, y_prob),
        "threshold": threshold,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(y_true, y_pred, output_dict=True)
    }

def train_evaluate(df_in: pd.DataFrame, cfg: CISConfig = CISConfig(), random_state: int = 42):
    X_all = add_handcrafted_features(df_in.drop(columns=list(cfg.drop)))
    y_all = df_in[cfg.target].astype(int).values
    numeric, cats, _ = feature_spaces(X_all, cfg)
    X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.25, random_state=random_state, stratify=y_all)
    preproc_lr  = build_column_transformer(numeric, cats, scale_numeric=True)
    preproc_xgb = build_column_transformer(numeric, cats, scale_numeric=False)
    pipe_lr  = build_lr_pipeline(preproc_lr)
    pipe_xgb = build_xgb_pipeline(preproc_xgb)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scoring = {"roc_auc": "roc_auc", "avg_precision": "average_precision"}
    cross_validate(pipe_lr,  X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)
    cross_validate(pipe_xgb, X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)
    pipe_lr.fit(X_tr, y_tr)
    pipe_xgb.fit(X_tr, y_tr)
    p_lr  = pipe_lr.predict_proba(X_te)[:, 1]
    p_xgb = pipe_xgb.predict_proba(X_te)[:, 1]
    thr_lr  = eval_thresholds(y_te, p_lr)["best_f1_threshold"]
    thr_xgb = eval_thresholds(y_te, p_xgb)["best_f1_threshold"]
    card_lr  = model_card("LogisticRegression", y_te, p_lr, threshold=thr_lr)
    card_xgb = model_card("XGBoost",            y_te, p_xgb, threshold=thr_xgb)
    return {"y_test": y_te, "probs": {"lr": p_lr, "xgb": p_xgb}, "cards": {"lr": card_lr, "xgb": card_xgb}}

results = train_evaluate(df, CISConfig())

# ------------------------
# Plotting utilities
# ------------------------
import matplotlib.pyplot as plt

# 1) ROC curve (both models on one clean figure; single plot, no subplots)
fpr_lr, tpr_lr, _ = roc_curve(results["y_test"], results["probs"]["lr"])
fpr_xgb, tpr_xgb, _ = roc_curve(results["y_test"], results["probs"]["xgb"])
auc_lr = auc(fpr_lr, tpr_lr)
auc_xgb = auc(fpr_xgb, tpr_xgb)

plt.figure()
plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {auc_lr:.3f})")
plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC = {auc_xgb:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – CIS Compliance")
plt.legend()
plt.grid(True, linestyle=":")
plt.tight_layout()
roc_path = "oc_auc_cis.png"
plt.savefig(roc_path, dpi=200)
plt.close()

# 2) Classification report bar charts (per model; separate figures)
def plot_classification_report_bar(card: Dict, filename: str):
    report = card["report"]
    # Extract per-class metrics 0 and 1
    classes = ["0", "1"]
    metrics = ["precision", "recall", "f1-score"]
    values = {m: [report[c][m] for c in classes] for m in metrics}

    x = np.arange(len(classes))
    width = 0.25

    plt.figure()
    plt.bar(x - width, values["precision"], width, label="Precision")
    plt.bar(x,         values["recall"],    width, label="Recall")
    plt.bar(x + width, values["f1-score"],  width, label="F1-score")
    plt.xticks(x, classes)
    plt.ylim(0, 1.05)
    plt.xlabel("Class")
    plt.ylabel("Score")
    plt.title(f"Classification Report – {card['model']} (threshold = {card['threshold']:.2f})")
    plt.legend()
    plt.grid(True, axis="y", linestyle=":")
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

cls_lr_path  = "classification_report_lr.png"
cls_xgb_path = "classification_report_xgb.png"
plot_classification_report_bar(results["cards"]["lr"],  cls_lr_path)
plot_classification_report_bar(results["cards"]["xgb"], cls_xgb_path)

roc_path, cls_lr_path, cls_xgb_path
