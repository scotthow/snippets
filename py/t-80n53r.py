import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, RFE, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')


class HEDISPreprocessor:
    """
    Preprocessor for HEDIS compliance prediction 
    """
    
    def __init__(self, target_col='numercnt', random_state=42):
        self.target_col = target_col
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_names = None
        self.selected_features = None
        self._fitted_states = None
        self._state_mapping = None
    
    def _log_transform_sdoh(self, df):
        """Apply a log transformation to skeweed SDoH features to handle non-normal distributions."""
        df_copy = df.copy()
        sdoh_cols = [
            "food_access", "health_access", "housing_desert", "poverty_score",
            "transport_access", "education_index", "citizenship_index",
            "housing_ownership", "income_index", "income_inequality",
            "natural_disaster", "social_isolation", "technology_access",
            "unemployment_index", "water_quality", "health_infra",
            "social_risk_score"
        ]
        
        for col in sdoh_cols:
            if col in df_copy.columns:
                df_copy[col] = np.log1p(df_copy[col])
        
        return df_copy

    def create_age_features(self, df):
      """
      Creates new features from the 'age' column to capture non-linear relationships
      and highlight potentially influential age groups.
      """
      df_copy = df.copy()

      # Create a polynomial feature to help linear models capture non-linear trends with age.
      df_copy['age_squared'] = df_copy['age'] ** 2
      
      # Create a binary flag for the senior age group. Senior populations often exhibit
      # distinct healthcare compliance behaviors. This assumes the highest value in the
      # categorical 'age' column represents the senior group.
      if not df_copy['age'].empty:
          senior_age_category = df_copy['age'].max()
          df_copy['is_senior'] = (df_copy['age'] == senior_age_category).astype(int)
      else:
          df_copy['is_senior'] = 0

      return df_copy

    
    def create_sdoh_composite_scores(self, df):
        df_copy = df.copy()
        df_copy['overall_access_score'] = (df_copy['food_access'] + df_copy['health_access'] + df_copy['transport_access'] + df_copy['technology_access']) / 4
        df_copy['vulnerability_score'] = (df_copy['poverty_score'] + df_copy['housing_desert'] + df_copy['social_isolation'] + df_copy['unemployment_index'] * 5) / 4
        df_copy['stability_score'] = (df_copy['housing_ownership'] + df_copy['income_index'] + df_copy['education_index'] + df_copy['citizenship_index']) / 4
        df_copy['environmental_risk'] = (df_copy['natural_disaster'] + (1 - df_copy['water_quality']) * 100) / 2
        df_copy['health_infra_vs_risk'] = (df_copy['health_infra'] / (df_copy['social_risk_score'] + 1))
        return df_copy
    
    def create_interaction_features(self, df):
        df_copy = df.copy()
        df_copy['age_health_access'] = df_copy['age'] * df_copy['health_access']
        df_copy['age_transport_access'] = df_copy['age'] * df_copy['transport_access']
        df_copy['poverty_health_access'] = df_copy['poverty_score'] * df_copy['health_access']
        df_copy['education_technology'] = df_copy['education_index'] * df_copy['technology_access']
        df_copy['inequality_health_impact'] = (df_copy['income_inequality'] * df_copy['health_access'])
        return df_copy

    def create_race_ethnicity_features(self, df):
        df_copy = df.copy()
        race_counts = df_copy['racetype'].value_counts()
        df_copy['race_grouped'] = df_copy['racetype'].apply(lambda x: x if race_counts.get(x, 0) >= 5 else 0)
        ethnicity_counts = df_copy['ethnicitytype'].value_counts()
        df_copy['ethnicity_grouped'] = df_copy['ethnicitytype'].apply(lambda x: x if ethnicity_counts.get(x, 0) >= 5 else 0)
        return df_copy
    
    def fit_transform(self, X, y):
        X_processed = X.copy()
        X_processed = self._log_transform_sdoh(X_processed)
        X_processed = self.create_age_features(X_processed)
        X_processed = self.create_sdoh_composite_scores(X_processed)
        X_processed = self.create_interaction_features(X_processed)
        X_processed = self.create_race_ethnicity_features(X_processed)
        
        self.feature_names = X_processed.columns.tolist()
        X_scaled = self.scaler.fit_transform(X_processed)
        return pd.DataFrame(X_scaled, columns=self.feature_names, index=X_processed.index)
    
    def transform(self, X):
        X_processed = X.copy()
        X_processed = self._log_transform_sdoh(X_processed)
        X_processed = self.create_age_features(X_processed)
        X_processed = self.create_sdoh_composite_scores(X_processed)
        X_processed = self.create_interaction_features(X_processed)
        X_processed = self.create_race_ethnicity_features(X_processed)
        
        X_scaled = self.scaler.transform(X_processed)
        return pd.DataFrame(X_scaled, columns=self.feature_names, index=X_processed.index)


class FeatureSelector:
    """
    Feature selection for HEDIS compliance prediction.
    """
    
    def __init__(self, method='hybrid', n_features=20, random_state=42):
        self.method = method
        self.n_features = n_features
        self.random_state = random_state
        self.selected_features = None
        self.feature_scores = None
        
    def select_features(self, X, y):
        if self.method == 'hybrid':
            return self._hybrid_selection(X, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
    def _hybrid_selection(self, X, y):
        n_features = X.shape[1]
        
        try:
            selector_f = SelectKBest(f_classif, k='all')
            selector_f.fit(X, y)
            f_scores = np.nan_to_num(selector_f.scores_, nan=0.0, posinf=0.0, neginf=0.0)
        except:
            f_scores = np.zeros(n_features)
        
        try:
            mi_scores = mutual_info_classif(X, y, random_state=self.random_state)
            mi_scores = np.nan_to_num(mi_scores, nan=0.0, posinf=0.0, neginf=0.0)
        except:
            mi_scores = np.zeros(n_features)
        
        try:
            rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state, class_weight='balanced', max_depth=5)
            rf.fit(X, y)
            rf_importance = rf.feature_importances_
        except:
            rf_importance = np.zeros(n_features)
        
        def safe_normalize(scores):
            max_val = scores.max()
            return scores / max_val if max_val > 0 else scores
        
        combined_scores = (0.3 * safe_normalize(f_scores) + 0.3 * safe_normalize(mi_scores) + 0.4 * safe_normalize(rf_importance))
        
        if combined_scores.max() == 0:
            variances = X.var()
            combined_scores = variances.values / (variances.max() + 1e-10)
        
        top_indices = np.argsort(combined_scores)[-self.n_features:]
        self.selected_features = X.columns[top_indices].tolist()
        self.feature_scores = pd.Series(combined_scores, index=X.columns).sort_values(ascending=False).fillna(0)
        return X[self.selected_features]


class HEDISModelPipeline:
    """
    Complete pipeline for HEDIS compliance prediction with adaptive imbalance handling.
    """
    
    def __init__(self, model_type='xgboost', random_state=42, 
                 imbalance_threshold=1.5, smote_strategy=0.75):
        self.model_type = model_type
        self.random_state = random_state
        self.imbalance_threshold = imbalance_threshold
        self.smote_strategy = smote_strategy
        self.preprocessor = HEDISPreprocessor(random_state=random_state)
        self.feature_selector = FeatureSelector(method='hybrid', n_features=20, random_state=random_state)
        self.model = None
        self.pipeline = None
        
    def build_model(self, scale_pos_weight=1):
        if self.model_type == 'logistic':
            return LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=1000, class_weight='balanced', random_state=self.random_state)
        elif self.model_type == 'xgboost':
            return XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                                 scale_pos_weight=scale_pos_weight, random_state=self.random_state, eval_metric='logloss')
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X, y):
        X_processed = self.preprocessor.fit_transform(X, y)
        X_selected = self.feature_selector.select_features(X_processed, y)
        
        counts = y.value_counts()
        scale_pos_weight = 1
        apply_smote = False

        if len(counts) == 2:
            ratio = counts.iloc[0] / counts.iloc[1]
            if ratio >= self.imbalance_threshold:
                apply_smote = True
                scale_pos_weight = counts.loc[0] / counts.loc[1]
        
        self.model = self.build_model(scale_pos_weight=scale_pos_weight)
        
        if apply_smote:
            print("SMOTE sampling applied to the model pipeline.")
            smote = SMOTE(random_state=self.random_state, sampling_strategy=self.smote_strategy)
            self.pipeline = ImbPipeline([('smote', smote), ('model', self.model)])
        else:
            print("SMOTE not applied. Training model on original data.")
            self.pipeline = self.model
        
        self.pipeline.fit(X_selected, y)
        return self
    
    def predict(self, X):
        X_processed = self.preprocessor.transform(X)
        X_selected = X_processed[self.feature_selector.selected_features]
        return self.pipeline.predict(X_selected)
    
    def predict_proba(self, X):
        X_processed = self.preprocessor.transform(X)
        X_selected = X_processed[self.feature_selector.selected_features]
        return self.pipeline.predict_proba(X_selected)
    
    def get_feature_importance(self):
        return self.feature_selector.feature_scores


class ModelVisualizer:
    """
    Visualization utilities for HEDIS compliance prediction models.
    """
    
    def __init__(self, figsize=(12, 8), style='seaborn-v0_8-darkgrid', dpi=100):
        self.figsize = figsize
        self.style = style
        self.dpi = dpi
        plt.style.use(self.style)
        
    def plot_classification_report(self, y_true, y_pred, model_name, save_path=None):
        report = classification_report(y_true, y_pred, output_dict=True)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'Classification Performance - {model_name}', fontsize=16, fontweight='bold')
        
        class_metrics = {'Non-Compliant (0)': report['0'], 'Compliant (1)': report['1']}
        metrics_df = pd.DataFrame(class_metrics).T[['precision', 'recall', 'f1-score']]
        
        sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1, ax=axes[0], cbar_kws={'label': 'Score'})
        axes[0].set_title('Classification Metrics by Class', fontsize=12, pad=10)
        
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1], cbar_kws={'label': 'Count'})
        axes[1].set_title('Confusion Matrix', fontsize=12, pad=10)
        axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual')
        axes[1].set_xticklabels(['Non-Compliant', 'Compliant']); axes[1].set_yticklabels(['Non-Compliant', 'Compliant'], rotation=0)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save_path: plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.show()
        
    def plot_roc_curves(self, models_data, save_path=None):
        fig, ax = plt.subplots(figsize=self.figsize)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for idx, (model_name, y_true, y_proba) in enumerate(models_data):
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[idx % len(colors)], lw=2.5, label=f'{model_name} (AUC = {roc_auc:.3f})', alpha=0.8)
        
        ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier (AUC = 0.500)')
        ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('False Positive Rate', fontsize=12); ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves - HEDIS Compliance Prediction Models', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.show()

    def plot_feature_importance(self, feature_scores, model_name, top_n=15, save_path=None):
        fig, ax = plt.subplots(figsize=(10, 8))
        top_features = feature_scores.head(top_n)
        colors = plt.cm.viridis(np.linspace(0.4, 0.9, len(top_features)))
        bars = ax.barh(range(len(top_features)), top_features.values, color=colors)
        ax.set_yticks(range(len(top_features))); ax.set_yticklabels(top_features.index)
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Top {top_n} Features - {model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.show()


class ModelExperimentRunner:
    """
    Runs a two-stage modeling experiment: a baseline test followed by a corrective run if needed.
    """
    def __init__(self, model_type='xgboost', roc_auc_threshold=0.75, random_state=42):
        self.model_type = model_type
        self.roc_auc_threshold = roc_auc_threshold
        self.random_state = random_state
        self.visualizer = ModelVisualizer(dpi=150)

    def run(self, X_train, y_train, X_test, y_test):
        """
        Executes the test-and-correct workflow.
        """
        # --- STAGE 1: BASELINE MODEL (TEST RUN) ---
        print("--- STAGE 1: Training Baseline Model (SMOTE is disabled) ---")
        # To disable SMOTE, we set the imbalance threshold to infinity.
        # The adaptive scale_pos_weight for XGBoost will still be calculated and used.
        baseline_pipeline = HEDISModelPipeline(
            model_type=self.model_type,
            random_state=self.random_state,
            imbalance_threshold=float('inf') 
        )
        baseline_pipeline.fit(X_train, y_train)
        
        y_pred_base = baseline_pipeline.predict(X_test)
        y_proba_base = baseline_pipeline.predict_proba(X_test)
        baseline_auc = roc_auc_score(y_test, y_proba_base[:, 1])

        print(f"\nBaseline Model ROC-AUC Score: {baseline_auc:.4f}")
        self.visualizer.plot_classification_report(y_test, y_pred_base, 'Baseline Model')
        self.visualizer.plot_feature_importance(baseline_pipeline.get_feature_importance(), 'Baseline Model')
        
        # --- DECISION POINT ---
        if baseline_auc >= self.roc_auc_threshold:
            print(f"\n✅ Baseline performance ({baseline_auc:.4f}) meets the threshold ({self.roc_auc_threshold}). No corrective SMOTE run needed.")
            self.visualizer.plot_roc_curves([('Baseline Model', y_test, y_proba_base)])
            return

        print(f"\n⚠️ Baseline performance ({baseline_auc:.4f}) is below the threshold ({self.roc_auc_threshold}).")
        
        # --- STAGE 2: CORRECTED MODEL ---
        print("\n--- STAGE 2: Training Corrected Model (SMOTE is enabled) ---")
        # Use the default threshold to allow SMOTE if the data is imbalanced
        corrected_pipeline = HEDISModelPipeline(
            model_type=self.model_type,
            random_state=self.random_state,
            imbalance_threshold=1.5 # Default value that allows SMOTE
        )
        corrected_pipeline.fit(X_train, y_train)
        
        y_pred_corr = corrected_pipeline.predict(X_test)
        y_proba_corr = corrected_pipeline.predict_proba(X_test)
        corrected_auc = roc_auc_score(y_test, y_proba_corr[:, 1])

        print(f"\nCorrected Model ROC-AUC Score: {corrected_auc:.4f}")
        print(f"Improvement over baseline: {corrected_auc - baseline_auc:+.4f}")
        self.visualizer.plot_classification_report(y_test, y_pred_corr, 'Corrected Model (SMOTE)')
        self.visualizer.plot_feature_importance(corrected_pipeline.get_feature_importance(), 'Corrected Model (SMOTE)')
        
        # Final comparative visualization
        print("\n--- Final Model Comparison ---")
        self.visualizer.plot_roc_curves([
            ('Baseline Model', y_test, y_proba_base),
            ('Corrected Model (w/ SMOTE)', y_test, y_proba_corr)
        ])


# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    np.random.seed(42)
    n = 1000
    
    # Generate imbalanced data for the primary test case
    data = {
        "age": np.random.randint(0, 3, n), "racetype": np.random.randint(1, 5, n),
        "ethnicitytype": np.random.randint(1, 25, n), "numercnt": np.random.choice([0, 1], size=n, p=[0.85, 0.15]),
        "food_access": np.random.uniform(0, 100, n), "health_access": np.random.uniform(0, 100, n),
        "housing_desert": np.random.uniform(0, 50, n), "poverty_score": np.random.uniform(0, 100, n),
        "transport_access": np.random.uniform(0, 100, n), "education_index": np.random.uniform(0, 100, n),
        "citizenship_index": np.random.uniform(80, 100, n), "housing_ownership": np.random.uniform(0, 100, n),
        "income_index": np.random.uniform(0, 100, n), "income_inequality": np.random.uniform(0, 100, n),
        "natural_disaster": np.random.uniform(0, 100, n), "social_isolation": np.random.uniform(0, 100, n),
        "technology_access": np.random.uniform(0, 100, n), "unemployment_index": np.random.uniform(0, 20, n),
        "water_quality": np.random.uniform(0, 1, n), "health_infra": np.random.uniform(0, 100, n),
        "social_risk_score": np.random.uniform(0, 100, n)
    }
    
    df = pd.DataFrame(data)

    X = df.drop('numercnt', axis=1)
    y = df['numercnt']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Initialize and run the experiment
    # The runner will handle the logic of when to apply SMOTE based on the 0.75 AUC threshold
    experiment = ModelExperimentRunner(model_type='xgboost', roc_auc_threshold=0.75)
    experiment.run(X_train, y_train, X_test, y_test)

