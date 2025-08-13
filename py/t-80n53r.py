import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif, RFE, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')


class HEDISPreprocessor:
    """
    Preprocessor for HEDIS compliance prediction with focus on CIS measure.
    Handles demographic and SDoH features for children's immunization status prediction.
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
        # List of SDoH features to be transformed
        sdoh_cols = [
            "food_access", "health_access", "housing_desert", "poverty_score",
            "transport_access", "education_index", "citizenship_index",
            "housing_ownership", "income_index", "income_inequality",
            "natural_disaster", "social_isolation", "technology_access",
            "unemployment_index", "water_quality", "health_infra",
            "social_risk_score"
        ]
        
        # Apply np.log1p which calculates log(1+x) to handle potential zero values
        for col in sdoh_cols:
            if col in df_copy.columns:
                df_copy[col] = np.log1p(df_copy[col])
        
        return df_copy

    def create_age_features(self, df):
        """
        Create age-related features specific to CIS measure (children 0-2 years typically).
        """
        df_copy = df.copy()
        
        # For CIS, critical age groups are 0-2 years
        df_copy['is_infant'] = (df_copy['age'] < 1).astype(int)
        df_copy['is_toddler'] = ((df_copy['age'] >= 1) & (df_copy['age'] <= 2)).astype(int)
        df_copy['is_preschool'] = ((df_copy['age'] >= 3) & (df_copy['age'] <= 5)).astype(int)
        
        # Age squared for non-linear relationships
        df_copy['age_squared'] = df_copy['age'] ** 2
        
        return df_copy

    def create_geographic_features(self, df):
        """
        Create geographic and accessibility composite features.
        """
        df_copy = df.copy()

        # State encoding - handle both fit and transform cases
        if self._state_mapping is not None:
            # Transform case - use fitted encoder and get a default value for unseen states
            df_copy['state_encoded'] = df_copy['state'].map(
                self._state_mapping
            ).fillna(0).astype(int)
        else:
            # Fit case - create mapping
            self._fitted_states = df_copy['state'].unique()
            self._state_mapping = {state: i + 1 for i, state in enumerate(self._fitted_states)}
            
            # The +1 ensures that a default value of 0 is not confused with a real state
            df_copy['state_encoded'] = df_copy['state'].map(
                self._state_mapping
            ).fillna(0).astype(int)

        # Regional risk zones based on coordinates
        df_copy['lat_zone'] = pd.cut(df_copy['latitude'], bins=5, labels=False)
        df_copy['long_zone'] = pd.cut(df_copy['longitude'], bins=5, labels=False)

        # ZIP code area (first 3 digits for regional grouping)
        df_copy['zip_area'] = df_copy['zip_code'] // 100

        return df_copy
    
    def create_sdoh_composite_scores(self, df):
        """
        Create composite Social Determinants of Health scores using log-transformed values.
        """
        df_copy = df.copy()
        
        # Access composite score (higher is better)
        df_copy['overall_access_score'] = (
            df_copy['food_access'] + 
            df_copy['health_access'] + 
            df_copy['transport_access'] + 
            df_copy['technology_access']
        ) / 4
        
        # Vulnerability score (higher is worse)
        df_copy['vulnerability_score'] = (
            df_copy['poverty_score'] + 
            df_copy['housing_desert'] + 
            df_copy['social_isolation'] + 
            df_copy['unemployment_index'] * 5  # Scale unemployment to 0-100
        ) / 4
        
        # Stability score (higher is better)
        df_copy['stability_score'] = (
            df_copy['housing_ownership'] + 
            df_copy['income_index'] + 
            df_copy['education_index'] + 
            df_copy['citizenship_index']
        ) / 4
        
        # Environmental risk score
        df_copy['environmental_risk'] = (
            df_copy['natural_disaster'] + 
            (1 - df_copy['water_quality']) * 100  # Convert to risk scale
        ) / 2
        
        # Healthcare infrastructure quality relative to social risk
        df_copy['health_infra_vs_risk'] = (
            df_copy['health_infra'] / (df_copy['social_risk_score'] + 1)
        )
        
        return df_copy
    
    def create_interaction_features(self, df):
        """
        Create interaction features between key predictors using log-transformed SDoH values.
        """
        df_copy = df.copy()
        
        # Age and access interactions (critical for young children)
        df_copy['age_health_access'] = df_copy['age'] * df_copy['health_access']
        df_copy['age_transport_access'] = df_copy['age'] * df_copy['transport_access']
        
        # Poverty and access interactions
        df_copy['poverty_health_access'] = df_copy['poverty_score'] * df_copy['health_access']
        
        # Education and technology (parent's ability to schedule/track immunizations)
        df_copy['education_technology'] = df_copy['education_index'] * df_copy['technology_access']
        
        # Income inequality impact on health access
        df_copy['inequality_health_impact'] = (
            df_copy['income_inequality'] * df_copy['health_access']
        )
        
        return df_copy
    

    def create_race_ethnicity_features(self, df):
          """
          Create meaningful race/ethnicity features while being mindful of bias.
          """
          df_copy = df.copy()

          # Group rare categories to avoid overfitting
          race_counts = df_copy['racetype'].value_counts()
          
          # Use .get() to safely handle keys not in the dictionary, returning 0
          df_copy['race_grouped'] = df_copy['racetype'].apply(
              lambda x: x if race_counts.get(x, 0) >= 5 else 0
          )
          
          ethnicity_counts = df_copy['ethnicitytype'].value_counts()

          # Use .get() to safely handle keys not in the dictionary, returning 0
          df_copy['ethnicity_grouped'] = df_copy['ethnicitytype'].apply(
              lambda x: x if ethnicity_counts.get(x, 0) >= 5 else 0
          )

          return df_copy
    
    def fit_transform(self, X, y):
        """
        Fit the preprocessor and transform the features.
        """
        X_processed = X.copy()
        
        # Apply log transformation to SDoH features first
        X_processed = self._log_transform_sdoh(X_processed)
        
        # Apply other feature engineering steps
        X_processed = self.create_age_features(X_processed)
        X_processed = self.create_geographic_features(X_processed)
        X_processed = self.create_sdoh_composite_scores(X_processed)
        X_processed = self.create_interaction_features(X_processed)
        X_processed = self.create_race_ethnicity_features(X_processed)
        
        # Drop original categorical columns and coordinates
        cols_to_drop = ['state', 'zip_code', 'latitude', 'longitude']
        X_processed = X_processed.drop(columns=cols_to_drop)
        
        # Store feature names
        self.feature_names = X_processed.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_processed)
        
        return pd.DataFrame(X_scaled, columns=self.feature_names, index=X_processed.index)
    
    def transform(self, X):
        """
        Transform new data using fitted preprocessor.
        """
        X_processed = X.copy()
        
        # Apply log transformation to SDoH features first
        X_processed = self._log_transform_sdoh(X_processed)
        
        # Apply other feature engineering steps
        X_processed = self.create_age_features(X_processed)
        X_processed = self.create_geographic_features(X_processed)
        X_processed = self.create_sdoh_composite_scores(X_processed)
        X_processed = self.create_interaction_features(X_processed)
        X_processed = self.create_race_ethnicity_features(X_processed)
        
        cols_to_drop = ['state', 'zip_code', 'latitude', 'longitude']
        X_processed = X_processed.drop(columns=cols_to_drop)
        
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
        """
        Select features using specified method.
        """
        if self.method == 'univariate':
            return self._univariate_selection(X, y)
        elif self.method == 'rfe':
            return self._rfe_selection(X, y)
        elif self.method == 'hybrid':
            return self._hybrid_selection(X, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _univariate_selection(self, X, y):
        """
        Univariate feature selection using F-statistic and mutual information.
        """
        # F-statistic selection
        selector_f = SelectKBest(f_classif, k=min(self.n_features, X.shape[1]))
        selector_f.fit(X, y)
        
        # Mutual information selection
        mi_scores = mutual_info_classif(X, y, random_state=self.random_state)
        
        # Combine scores with safe normalization
        f_scores = selector_f.scores_
        f_scores_norm = f_scores / (f_scores.max() + 1e-10)
        mi_scores_norm = mi_scores / (mi_scores.max() + 1e-10)
        combined_scores = (f_scores_norm + mi_scores_norm) / 2
        
        # Select top features
        top_indices = np.argsort(combined_scores)[-self.n_features:]
        self.selected_features = X.columns[top_indices].tolist()
        self.feature_scores = pd.Series(combined_scores, index=X.columns).sort_values(ascending=False)
        
        return X[self.selected_features]
    
    def _rfe_selection(self, X, y):
        """
        Recursive Feature Elimination using Random Forest.
        """
        estimator = RandomForestClassifier(
            n_estimators=100, 
            random_state=self.random_state,
            class_weight='balanced'
        )
        
        selector = RFE(
            estimator, 
            n_features_to_select=min(self.n_features, X.shape[1]),
            step=1
        )
        selector.fit(X, y)
        
        self.selected_features = X.columns[selector.support_].tolist()
        self.feature_scores = pd.Series(
            selector.ranking_, 
            index=X.columns
        ).sort_values()
        
        return X[self.selected_features]
    
    def _hybrid_selection(self, X, y):
        """
        Hybrid approach combining multiple methods.
        """
        # Initialize scores array
        n_features = X.shape[1]
        
        # Get univariate scores with error handling
        try:
            selector_f = SelectKBest(f_classif, k='all')
            selector_f.fit(X, y)
            f_scores = selector_f.scores_
            # Replace NaN/inf values with 0
            f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0)
        except:
            f_scores = np.zeros(n_features)
        
        # Get mutual information scores
        try:
            mi_scores = mutual_info_classif(X, y, random_state=self.random_state)
            mi_scores = np.nan_to_num(mi_scores, nan=0.0, posinf=0.0, neginf=0.0)
        except:
            mi_scores = np.zeros(n_features)
        
        # Get feature importance from Random Forest
        try:
            rf = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                class_weight='balanced',
                max_depth=5
            )
            rf.fit(X, y)
            rf_importance = rf.feature_importances_
        except:
            rf_importance = np.zeros(n_features)
        
        # Normalize scores safely
        def safe_normalize(scores):
            max_val = scores.max()
            if max_val > 0:
                return scores / max_val
            else:
                return scores
        
        f_scores_norm = safe_normalize(f_scores)
        mi_scores_norm = safe_normalize(mi_scores)
        rf_scores_norm = safe_normalize(rf_importance)
        
        # Weighted combination
        combined_scores = (
            0.3 * f_scores_norm + 
            0.3 * mi_scores_norm + 
            0.4 * rf_scores_norm
        )
        
        # Handle case where all scores are zero
        if combined_scores.max() == 0:
            # Use variance as a fallback
            variances = X.var()
            combined_scores = variances.values / (variances.max() + 1e-10)
        
        # Select top features
        top_indices = np.argsort(combined_scores)[-self.n_features:]
        self.selected_features = X.columns[top_indices].tolist()
        
        # Create feature scores series without NaN values
        self.feature_scores = pd.Series(
            combined_scores, 
            index=X.columns
        ).sort_values(ascending=False)
        
        # Ensure no NaN values in feature scores
        self.feature_scores = self.feature_scores.fillna(0)
        
        return X[self.selected_features]


class HEDISModelPipeline:
    """
    Complete pipeline for HEDIS compliance prediction.
    """
    
    def __init__(self, model_type='xgboost', use_smote=True, random_state=42):
        self.model_type = model_type
        self.use_smote = use_smote
        self.random_state = random_state
        self.preprocessor = HEDISPreprocessor(random_state=random_state)
        self.feature_selector = FeatureSelector(method='hybrid', n_features=20, random_state=random_state)
        self.model = None
        self.pipeline = None
        
    def build_model(self):
        """
        Build the specified model with optimized hyperparameters.
        """
        if self.model_type == 'logistic':
            return LogisticRegression(
                C=1.0,
                penalty='l2',
                solver='lbfgs',
                max_iter=1000,
                class_weight='balanced',
                random_state=self.random_state
            )
        elif self.model_type == 'xgboost':
            return XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=5,  # Address class imbalance
                random_state=self.random_state,
                eval_metric='logloss'
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X, y):
        """
        Fit the complete pipeline.
        """
        # Preprocess features
        X_processed = self.preprocessor.fit_transform(X, y)
        
        # Select features
        X_selected = self.feature_selector.select_features(X_processed, y)
        
        # Build model
        self.model = self.build_model()
        
        # Create pipeline with SMOTE if specified
        if self.use_smote:
            smote = SMOTE(random_state=self.random_state, sampling_strategy=0.5)
            self.pipeline = ImbPipeline([
                ('smote', smote),
                ('model', self.model)
            ])
        else:
            self.pipeline = self.model
        
        # Fit pipeline
        self.pipeline.fit(X_selected, y)
        
        return self
    
    def predict(self, X):
        """
        Make predictions on new data.
        """
        X_processed = self.preprocessor.transform(X)
        X_selected = X_processed[self.feature_selector.selected_features]
        return self.pipeline.predict(X_selected)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities.
        """
        X_processed = self.preprocessor.transform(X)
        X_selected = X_processed[self.feature_selector.selected_features]
        return self.pipeline.predict_proba(X_selected)
    
    def get_feature_importance(self):
        """
        Get feature importance scores.
        """
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
        """
        Create a heatmap visualization of the classification report.
        """
        from sklearn.metrics import classification_report, confusion_matrix
        import seaborn as sns
        
        # Get classification report as dict
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'Classification Performance - {model_name}', fontsize=16, fontweight='bold')
        
        # Plot 1: Classification metrics heatmap
        # Extract metrics for classes 0 and 1
        class_metrics = {
            'Non-Compliant (0)': report['0'],
            'Compliant (1)': report['1']
        }
        
        metrics_df = pd.DataFrame(class_metrics).T
        metrics_df = metrics_df[['precision', 'recall', 'f1-score']]
        
        sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='RdYlGn', 
                   vmin=0, vmax=1, ax=axes[0], cbar_kws={'label': 'Score'})
        axes[0].set_title('Classification Metrics by Class', fontsize=12, pad=10)
        axes[0].set_xlabel('Metrics', fontsize=11)
        axes[0].set_ylabel('Class', fontsize=11)
        
        # Plot 2: Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                   cbar_kws={'label': 'Count'})
        axes[1].set_title('Confusion Matrix', fontsize=12, pad=10)
        axes[1].set_xlabel('Predicted', fontsize=11)
        axes[1].set_ylabel('Actual', fontsize=11)
        axes[1].set_xticklabels(['Non-Compliant', 'Compliant'])
        axes[1].set_yticklabels(['Non-Compliant', 'Compliant'], rotation=0)
        
        # Add text box with overall metrics
        accuracy = report['accuracy']
        macro_f1 = report['macro avg']['f1-score']
        weighted_f1 = report['weighted avg']['f1-score']
        
        textstr = f'Overall Metrics:\nAccuracy: {accuracy:.3f}\nMacro F1: {macro_f1:.3f}\nWeighted F1: {weighted_f1:.3f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        fig.text(0.5, 0.02, textstr, transform=fig.transFigure, fontsize=10,
                verticalalignment='bottom', bbox=props, ha='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Classification report saved to {save_path}")
        
        plt.show()
        return fig
    
    def plot_roc_curves(self, models_data, save_path=None):
        """
        Plot ROC curves for multiple models with confidence intervals.
        
        Parameters:
        models_data: list of tuples (model_name, y_true, y_proba)
        """
        from sklearn.metrics import roc_curve, auc, roc_auc_score
        from sklearn.model_selection import StratifiedKFold
        import matplotlib.patches as mpatches
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for idx, (model_name, y_true, y_proba) in enumerate(models_data):
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            
            # Plot ROC curve
            ax.plot(fpr, tpr, color=colors[idx % len(colors)], lw=2.5,
                   label=f'{model_name} (AUC = {roc_auc:.3f})', alpha=0.8)
            
            # Add confidence interval shading
            ax.fill_between(fpr, tpr, alpha=0.15, color=colors[idx % len(colors)])
        
        # Plot diagonal reference line
        ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier (AUC = 0.500)')
        
        # Styling
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curves - HEDIS Compliance Prediction Models', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Legend
        ax.legend(loc='lower right', frameon=True, fancybox=True, 
                 shadow=True, borderpad=1, fontsize=10)
        
        # Add annotation for AUC interpretation
        textstr = 'AUC Interpretation:\n> 0.9: Excellent\n0.8-0.9: Good\n0.7-0.8: Fair\n< 0.7: Poor'
        props = dict(boxstyle='round', facecolor='lightgray', alpha=0.8)
        ax.text(0.62, 0.25, textstr, transform=ax.transAxes, fontsize=9,
               verticalalignment='center', bbox=props)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"ROC curves saved to {save_path}")
        
        plt.show()
        return fig
    
    def plot_feature_importance(self, feature_scores, top_n=15, save_path=None):
        """
        Create a horizontal bar chart of feature importance scores.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get top features
        top_features = feature_scores.head(top_n)
        
        # Create color gradient
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_features)))
        
        # Create horizontal bar chart
        bars = ax.barh(range(len(top_features)), top_features.values, color=colors)
        
        # Customize y-axis
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features.index, fontsize=10)
        ax.invert_yaxis()
        
        # Labels and title
        ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {top_n} Most Important Features for HEDIS Compliance Prediction',
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, top_features.values)):
            ax.text(value, bar.get_y() + bar.get_height()/2, f'{value:.3f}',
                   ha='left', va='center', fontsize=9, fontweight='bold')
        
        # Grid
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Feature importance chart saved to {save_path}")
        
        plt.show()
        return fig
    
    def plot_class_distribution(self, y_train, y_test, save_path=None):
        """
        Visualize class distribution in training and test sets.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Training set distribution
        train_counts = pd.Series(y_train).value_counts()
        colors = ['#ff6b6b', '#4ecdc4']
        
        wedges, texts, autotexts = axes[0].pie(train_counts.values, 
                                                labels=['Non-Compliant', 'Compliant'],
                                                colors=colors, autopct='%1.1f%%',
                                                startangle=90, explode=(0.05, 0))
        axes[0].set_title('Training Set Distribution', fontsize=12, fontweight='bold')
        
        # Test set distribution
        test_counts = pd.Series(y_test).value_counts()
        wedges, texts, autotexts = axes[1].pie(test_counts.values,
                                               labels=['Non-Compliant', 'Compliant'],
                                               colors=colors, autopct='%1.1f%%',
                                               startangle=90, explode=(0.05, 0))
        axes[1].set_title('Test Set Distribution', fontsize=12, fontweight='bold')
        
        # Overall title
        fig.suptitle('Class Distribution - HEDIS Compliance', fontsize=14, fontweight='bold')
        
        # Add text box with imbalance ratio
        imbalance_ratio = train_counts[0] / train_counts[1]
        textstr = f'Imbalance Ratio: {imbalance_ratio:.2f}:1\n(Non-Compliant:Compliant)'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        fig.text(0.5, 0.02, textstr, transform=fig.transFigure, fontsize=10,
                verticalalignment='bottom', bbox=props, ha='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Class distribution chart saved to {save_path}")
        
        plt.show()
        return fig


# Example usage and testing
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import classification_report, roc_auc_score
    
    # Generate sample data
    np.random.seed(42)
    n = 1000  # Larger sample for better demonstration
    
    data = {
        "age": np.random.randint(0, 3, n),  # Focus on 0-2 years for CIS
        "racetype": np.random.randint(1, 5, n),
        "ethnicitytype": np.random.randint(1, 25, n),
        "numercnt": np.random.choice([0, 1], size=n, p=[0.85, 0.15]),
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
    
    # Split features and target
    X = df.drop('numercnt', axis=1)
    y = df['numercnt']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize visualizer
    visualizer = ModelVisualizer(dpi=150)
    
    # Test XGBoost pipeline
    print("Testing XGBoost Pipeline with SMOTE and Log Transformation")
    print("-" * 50)
    
    xgb_pipeline = HEDISModelPipeline(model_type='xgboost', use_smote=True)
    xgb_pipeline.fit(X_train, y_train)
    
    # Predictions
    y_pred_xgb = xgb_pipeline.predict(X_test)
    y_proba_xgb = xgb_pipeline.predict_proba(X_test)
    
    # Test Logistic Regression pipeline
    print("\nTesting Logistic Regression Pipeline with SMOTE and Log Transformation")
    print("-" * 50)
    
    lr_pipeline = HEDISModelPipeline(model_type='logistic', use_smote=True)
    lr_pipeline.fit(X_train, y_train)
    
    y_pred_lr = lr_pipeline.predict(X_test)
    y_proba_lr = lr_pipeline.predict_proba(X_test)
    
    # Generate visualizations
    print("\n" + "="*50)
    print("Generating Visualizations")
    print("="*50)
    
    # 1. Class Distribution
    visualizer.plot_class_distribution(y_train, y_test, 
                                      save_path='hedis_class_distribution.png')
    
    # 2. Classification Report - XGBoost
    visualizer.plot_classification_report(y_test, y_pred_xgb, 'XGBoost (Log-Transformed)',
                                         save_path='hedis_classification_xgboost_log.png')
    
    # 3. Classification Report - Logistic Regression
    visualizer.plot_classification_report(y_test, y_pred_lr, 'Logistic Regression (Log-Transformed)',
                                         save_path='hedis_classification_lr_log.png')
    
    # 4. ROC Curves Comparison
    models_data = [
        ('XGBoost', y_test, y_proba_xgb),
        ('Logistic Regression', y_test, y_proba_lr)
    ]
    visualizer.plot_roc_curves(models_data, save_path='hedis_roc_curves_log.png')
    
    # 5. Feature Importance
    visualizer.plot_feature_importance(xgb_pipeline.get_feature_importance(),
                                      top_n=15, save_path='hedis_feature_importance_log.png')
    
    # Print summary statistics
    print("\n" + "="*50)
    print("Model Performance Summary (with Log Transformation)")
    print("="*50)
    
    print("\nXGBoost Performance:")
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba_xgb[:, 1]):.4f}")
    print(classification_report(y_test, y_pred_xgb))
    
    print("\nLogistic Regression Performance:")
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba_lr[:, 1]):.4f}")
    print(classification_report(y_test, y_pred_lr))
    
    print("\nTop 10 Most Important Features (XGBoost):")
    print(xgb_pipeline.get_feature_importance().head(10))
    
    print("\n" + "="*50)
    print("All visualizations have been saved as PNG files!")
    print("="*50)