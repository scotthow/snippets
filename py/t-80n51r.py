# core.py

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve
from sklearn.utils.validation import check_is_fitted
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier


# ----------------------------
# Column configuration
# ----------------------------
@dataclass
class ColumnConfig:
    # Columns known to be identifiers or high risk of leakage
    drop_cols: List[str] = field(default_factory=lambda: [
        # IDs / near-IDs
        "member_key", "src_mbr_id", "asdb_plankey", "indiv_anlytcs_id",
        "iodb_mbr_key", "iodb_plan_key", "cust_no", "cust_nm",
        # Addresses / high-cardinality geocodes that add little beyond SDoH
        "address_line_1", "city",
        # Strong leakage or not reliably available before outcome
        "clssdt", "max_source_pstd_dts",
        # Often too granular & redundant with SDoH
        "block_code", "bgfips",
    ])
    # Optionally forbidden to avoid leakage (toggle as needed)
    forbid_cols: List[str] = field(default_factory=lambda: [
        # Denominator indicates eligibility and can collapse the task
        "denomcnt",
    ])
    # Date-like string columns to parse (safe, non-leaky)
    date_cols: List[str] = field(default_factory=lambda: [
        "bus_eff_dt",
    ])
    # Categorical columns (object or coded as int but categorical)
    cat_cols: List[str] = field(default_factory=lambda: [
        "measure_tla", "submeasurekey",
        "cust_seg_cd", "lob_cd", "state",
        "racetype", "ethnicitytype", "zip_code",
    ])
    # Numeric continuous columns (keep as float64)
    sdoh_cols: List[str] = field(default_factory=lambda: [
        "food_access", "health_access", "housing_desert", "poverty_score",
        "transport_access", "education_index", "citizenship_index",
        "housing_ownership", "income_index", "income_inequality",
        "natural_disaster", "social_isolation", "technology_access",
        "unemployment_index", "water_quality", "health_infra",
        "social_risk_score", "latitude", "longitude",
    ])
    # Safe numeric columns
    num_cols: List[str] = field(default_factory=lambda: [
        "age", "measurement_year", "effective_year",
    ])

    def numeric_feature_list(self) -> List[str]:
        return list(dict.fromkeys(self.sdoh_cols + self.num_cols))


# ----------------------------
# Utilities / Transformers
# ----------------------------
class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, to_drop: List[str]):
        self.to_drop = list(dict.fromkeys(to_drop))

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        cols = [c for c in self.to_drop if c in X.columns]
        return X.drop(columns=cols, errors="ignore")


class DateParts(BaseEstimator, TransformerMixin):
    """
    Parses date columns (string/object) into float64 parts:
      - days_since_epoch
      - year, month, quarter
    """
    def __init__(self, date_cols: List[str]):
        self.date_cols = date_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.date_cols:
            if c in X.columns:
                d = pd.to_datetime(X[c], errors="coerce", utc=True)
                X[f"{c}_days"] = (d.view("int64") // 10**9) / 86400.0  # seconds->days
                X[f"{c}_year"] = d.dt.year.astype("float64")
                X[f"{c}_month"] = d.dt.month.astype("float64")
                X[f"{c}_quarter"] = d.dt.quarter.astype("float64")
        return X


class TenureFromEffectiveYear(BaseEstimator, TransformerMixin):
    """
    Tenure (years) from business effective date to measurement_year end (Dec 31).
    Requires 'bus_eff_dt' to have been parsed by DateParts (or still present as parsable string).
    """
    def __init__(self, eff_col: str = "bus_eff_dt", year_col: str = "measurement_year"):
        self.eff_col = eff_col
        self.year_col = year_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.eff_col in X.columns and self.year_col in X.columns:
            eff = pd.to_datetime(X[self.eff_col], errors="coerce", utc=True)
            yr = pd.to_numeric(X[self.year_col], errors="coerce").astype("Int64")
            end_dates = pd.to_datetime(
                yr.astype("float").astype("Int64").astype("string") + "-12-31", errors="coerce", utc=True
            )
            tenure_days = (end_dates - eff).dt.days.astype("float64")
            X["tenure_years"] = (tenure_days / 365.25).astype("float64")
        return X


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """
    Groups infrequent categories into 'OTHER' to stabilize encodings.
    """
    def __init__(self, cols: List[str], min_freq: int = 100, min_prop: float = 0.005):
        self.cols = cols
        self.min_freq = min_freq
        self.min_prop = min_prop
        self._kept_: Dict[str, set] = {}

    def fit(self, X, y=None):
        self._kept_ = {}
        n = len(X)
        for c in self.cols:
            if c in X.columns:
                vc = X[c].astype("object").value_counts(dropna=False)
                keep = set(vc[(vc >= self.min_freq) | (vc / max(n, 1) >= self.min_prop)].index.tolist())
                self._kept_[c] = keep
        return self

    def transform(self, X):
        check_is_fitted(self, "_kept_")
        X = X.copy()
        for c in self.cols:
            if c in X.columns:
                kept = self._kept_.get(c, set())
                X[c] = X[c].astype("object").where(X[c].isin(kept), other="__OTHER__")
        return X


class CVTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Out-of-fold target encoding with smoothing to avoid leakage.
    For binary target y in {0,1}.

    smoothing formula (per category):
        enc = (n * mean_cat + alpha * global_mean) / (n + alpha)

    Parameters
    ----------
    cols : list of categorical column names
    n_splits : int, CV folds
    alpha : float, smoothing strength
    random_state : int
    """
    def __init__(self, cols: List[str], n_splits: int = 5, alpha: float = 20.0, random_state: int = 42):
        self.cols = cols
        self.n_splits = n_splits
        self.alpha = alpha
        self.random_state = random_state
        self._global_mean_: float = 0.0
        self._maps_: Dict[str, pd.Series] = {}

    def fit(self, X, y):
        X = X.copy()
        y = pd.Series(y).astype(float).values
        self._global_mean_ = float(np.nanmean(y))
        self._maps_ = {c: pd.Series(dtype="float64") for c in self.cols}

        kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        oof_encs = {c: np.zeros(len(X), dtype="float64") for c in self.cols}

        # out-of-fold means
        for tr_idx, te_idx in kf.split(X, y):
            y_tr = y[tr_idx]
            gm = float(np.nanmean(y_tr))
            for c in self.cols:
                vals = X.iloc[tr_idx][c].astype("object")
                grp = pd.DataFrame({"cat": vals, "y": y_tr}).groupby("cat")["y"].agg(["mean", "count"])
                enc = (grp["count"] * grp["mean"] + self.alpha * gm) / (grp["count"] + self.alpha)
                oof_encs[c][te_idx] = X.iloc[te_idx][c].astype("object").map(enc).fillna(gm).astype("float64")

        # store full-data maps for transform
        for c in self.cols:
            vals = X[c].astype("object")
            grp = pd.DataFrame({"cat": vals, "y": y}).groupby("cat")["y"].agg(["mean", "count"])
            enc = (grp["count"] * grp["mean"] + self.alpha * self._global_mean_) / (grp["count"] + self.alpha)
            self._maps_[c] = enc.astype("float64")

        # keep the oof encodings for potential diagnostics (optional)
        self._oof_ = pd.DataFrame({c: oof_encs[c] for c in self.cols})
        return self

    def transform(self, X):
        check_is_fitted(self, "_maps_")
        X = X.copy()
        for c in self.cols:
            m = self._maps_.get(c, pd.Series(dtype="float64"))
            X[f"te__{c}"] = X[c].astype("object").map(m).fillna(self._global_mean_).astype("float64")
        return X


class NumericCleaner(BaseEstimator, TransformerMixin):
    """
    - Ensures numeric columns are float64
    - Optional winsorization (clipping) to reduce outlier influence
    - Imputes missing with median (imputation can also be done later in pipeline)
    """
    def __init__(self, num_cols: List[str], clip_quantiles: Tuple[float, float]=(0.01, 0.99), do_impute: bool=True):
        self.num_cols = num_cols
        self.clip_quantiles = clip_quantiles
        self.do_impute = do_impute
        self._medians_: Dict[str, float] = {}
        self._q_: Dict[str, Tuple[float,float]] = {}

    def fit(self, X, y=None):
        X = X.copy()
        self._medians_.clear()
        self._q_.clear()
        for c in self.num_cols:
            if c in X.columns:
                col = pd.to_numeric(X[c], errors="coerce")
                q_low, q_hi = col.quantile(self.clip_quantiles[0]), col.quantile(self.clip_quantiles[1])
                self._q_[c] = (float(q_low), float(q_hi))
                self._medians_[c] = float(col.median())
        return self

    def transform(self, X):
        check_is_fitted(self, "_medians_")
        X = X.copy()
        for c in self.num_cols:
            if c in X.columns:
                col = pd.to_numeric(X[c], errors="coerce").astype("float64")
                ql, qh = self._q_[c]
                col = col.clip(lower=ql, upper=qh)
                if self.do_impute:
                    col = col.fillna(self._medians_[c])
                X[c] = col.astype("float64")
        return X


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """
    Drops one variable from pairs with |corr| >= threshold.
    Works on float-only dataframes.
    """
    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        self._keep_: Optional[List[str]] = None

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        corr = X.corr(numeric_only=True).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop = [column for column in upper.columns if any(upper[column] >= self.threshold)]
        self._keep_ = [c for c in X.columns if c not in drop]
        return self

    def transform(self, X):
        check_is_fitted(self, "_keep_")
        X = pd.DataFrame(X).copy()
        return X[self._keep_]


# ----------------------------
# High-level feature builder
# ----------------------------
class HEDISFeatureBuilder(BaseEstimator, TransformerMixin):
    """
    End-to-end, leak-aware feature engineering that returns a float64 DataFrame.
    Steps:
      1) Drop leaky/ID columns
      2) Parse dates -> parts + tenure
      3) Group rare categories
      4) Target-encode categoricals (CV, smoothed)
      5) Clean numeric (clip + impute)
      6) Assemble float64 matrix
    """
    def __init__(
        self,
        cfg: ColumnConfig,
        forbid_denom: bool = True,
        rare_min_freq: int = 100,
        rare_min_prop: float = 0.005,
        te_alpha: float = 20.0,
        te_folds: int = 5,
        random_state: int = 42
    ):
        self.cfg = cfg
        self.forbid_denom = forbid_denom
        self.rare = RareCategoryGrouper(cfg.cat_cols, min_freq=rare_min_freq, min_prop=rare_min_prop)
        self.date_parts = DateParts(cfg.date_cols)
        self.tenure = TenureFromEffectiveYear()
        self.te = CVTargetEncoder(cfg.cat_cols, n_splits=te_folds, alpha=te_alpha, random_state=random_state)
        self.num_clean = NumericCleaner(cfg.numeric_feature_list())
        self.dropper = None  # initialized in fit

    def _dropper(self) -> ColumnDropper:
        drops = list(self.cfg.drop_cols)
        if self.forbid_denom:
            drops += self.cfg.forbid_cols
        return ColumnDropper(drops)

    def fit(self, X: pd.DataFrame, y):
        self.dropper = self._dropper()
        Xp = self.dropper.transform(X)
        Xp = self.date_parts.fit_transform(Xp, y)
        Xp = self.tenure.fit_transform(Xp, y)
        Xp = self.rare.fit_transform(Xp, y)
        Xp = self.te.fit_transform(Xp, y)  # needs y
        # ensure numeric cleaning covers base numeric + newly created TE columns
        extra_te_cols = [f"te__{c}" for c in self.cfg.cat_cols]
        self.num_clean.num_cols = list(dict.fromkeys(self.cfg.numeric_feature_list() +
                                                     extra_te_cols +
                                                     [f"{c}_days" for c in self.cfg.date_cols] +
                                                     [f"{c}_year" for c in self.cfg.date_cols] +
                                                     [f"{c}_month" for c in self.cfg.date_cols] +
                                                     [f"{c}_quarter" for c in self.cfg.date_cols] +
                                                     ["tenure_years"]))
        self.num_clean.fit(Xp, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "dropper")
        Xp = self.dropper.transform(X)
        Xp = self.date_parts.transform(Xp)
        Xp = self.tenure.transform(Xp)
        Xp = self.rare.transform(Xp)
        Xp = self.te.transform(Xp)
        Xp = self.num_clean.transform(Xp)

        # Select only numeric float64 columns
        numeric_cols = [c for c in Xp.columns if pd.api.types.is_numeric_dtype(Xp[c])]
        Xf = Xp[numeric_cols].astype("float64")
        # assure no inf/nan
        Xf = Xf.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float64")
        return Xf


# ----------------------------
# Pipeline builders
# ----------------------------
def build_pipelines(
    cfg: ColumnConfig,
    forbid_denom: bool = True,
    random_state: int = 42
) -> Dict[str, ImbPipeline]:
    """
    Returns two imblearn pipelines with SMOTE:
      - 'logreg': scaling + correlation filter + L1 feature selection + LogisticRegression
      - 'xgb':    correlation filter + XGBClassifier
    """
    feat = HEDISFeatureBuilder(cfg, forbid_denom=forbid_denom, random_state=random_state)

    # Logistic Regression branch
    logreg_selector = SelectFromModel(
        LogisticRegression(
            penalty="l1", solver="liblinear", C=0.5, max_iter=2000, random_state=random_state
        ),
        threshold="median", prefit=False
    )
    logreg = ImbPipeline(steps=[
        ("features", feat),
        ("imputer", SimpleImputer(strategy="median")),      # safety net
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("corr_filter", CorrelationFilter(threshold=0.95)),
        ("smote", SMOTE(random_state=random_state, k_neighbors=5)),
        ("sel_l1", logreg_selector),
        ("clf", LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", max_iter=5000, random_state=random_state
        )),
    ])

    # XGBoost branch
    xgb = ImbPipeline(steps=[
        ("features", feat),
        ("imputer", SimpleImputer(strategy="median")),
        ("corr_filter", CorrelationFilter(threshold=0.98)),
        ("smote", SMOTE(random_state=random_state, k_neighbors=5)),
        ("clf", XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            random_state=random_state
        )),
    ])

    return {"logreg": logreg, "xgb": xgb}


# ----------------------------
# Training / evaluation helpers
# ----------------------------
def optimal_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """
    Returns (best_threshold, best_f1) by scanning PR curve thresholds.
    """
    p, r, t = precision_recall_curve(y_true, y_prob)
    f1s = 2 * (p * r) / (p + r + 1e-12)
    idx = int(np.nanargmax(f1s))
    # precision_recall_curve returns thresholds of length len(p)-1
    thr = t[max(0, min(idx, len(t)-1))] if len(t) else 0.5
    return float(thr), float(np.nanmax(f1s))


def evaluate_pipeline(
    pipe: ImbPipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray
) -> Dict[str, float]:
    pipe.fit(X_train, y_train)
    prob = pipe.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, prob)
    ap = average_precision_score(y_valid, prob)
    thr, f1 = optimal_f1_threshold(y_valid, prob)
    y_pred = (prob >= thr).astype(int)
    return {"roc_auc": auc, "avg_prec": ap, "f1_opt": f1, "thr_opt": thr}


def prepare_data(df: pd.DataFrame, cfg: ColumnConfig) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Basic y extraction + defensive casting; assumes y is binary {0,1}.
    """
    y = pd.to_numeric(df["numercnt"], errors="coerce").fillna(0).clip(0, 1).astype(int).values
    X = df.drop(columns=["numercnt"], errors="ignore")
    return X, y


def train_and_compare(df: pd.DataFrame, forbid_denom: bool = True, random_state: int = 42):
    """
    Example end-to-end usage. Splits, trains both models, and prints metrics.
    """
    cfg = ColumnConfig()
    X, y = prepare_data(df, cfg)

    # temporal split is often better; here we stratify for simplicity
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=random_state
    )

    pipes = build_pipelines(cfg, forbid_denom=forbid_denom, random_state=random_state)

    results = {}
    for name, pipe in pipes.items():
        results[name] = evaluate_pipeline(pipe, X_tr, y_tr, X_va, y_va)

    # choose the better by avg precision (robust for imbalance)
    best_name = max(results, key=lambda k: results[k]["avg_prec"])
    best_pipe = pipes[best_name]

    return {
        "results": results,
        "best_model": best_name,
        "pipeline": best_pipe
    }
