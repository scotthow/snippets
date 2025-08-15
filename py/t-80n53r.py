
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import normaltest, skew
import warnings
import logging
import os
import sys
from datetime import datetime

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

def setup_logging(log_dir="output"):
    """
    Sets up logging to write to both a file and the console.
    A new timestamped log file is created in the specified directory for each run.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Generate a unique filename with a timestamp
    log_filename = os.path.join(log_dir, f"hedis_pipeline_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    # Configure the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear any existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a formatter for the log file (includes timestamp and level)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Create a handler to write to the log file
    file_handler = logging.FileHandler(log_filename, 'w', 'utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Create a handler to write to the console (simple message format)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logging.info(f"Logging initialized. Output will be saved to '{log_filename}'")


class HEDISPreprocessor:
    """
    Preprocessor for HEDIS compliance prediction data.
    Handles feature engineering, transformation, and selection.
    """
    
    def __init__(self, target_col='numercnt', id_cols=None):
        self.target_col = target_col
        self.id_cols = id_cols or ['src_mbr_id']
        self.categorical_cols = []
        self.numerical_cols = []
        
        # Specify columns for different encoding strategies
        self.one_hot_encode_cols = ['racetype', 'ethnicitytype']
        self.label_encode_cols = []
        self.one_hot_encoded_columns = []
        
        self.log_transform_cols = []
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance_scores = {}
        
    def identify_column_types(self, df):
        """Identify categorical and numerical columns and assign encoding strategy"""
        exclude_cols = self.id_cols + [self.target_col]
        
        self.categorical_cols = []
        self.numerical_cols = []
        self.label_encode_cols = []
        
        for col in df.columns:
            if col not in exclude_cols:
                if df[col].dtype == 'object' or col in ['sex', 'racetype', 'ethnicitytype']:
                    self.categorical_cols.append(col)
                    if col in self.one_hot_encode_cols:
                        continue 
                    else:
                        self.label_encode_cols.append(col)
                else:
                    self.numerical_cols.append(col)
        
        logging.info(f"Identified {len(self.categorical_cols)} total categorical columns")
        logging.info(f"  - To be one-hot encoded: {self.one_hot_encode_cols}")
        logging.info(f"  - To be label encoded: {self.label_encode_cols}")
        logging.info(f"Identified {len(self.numerical_cols)} numerical columns")
        
    def check_normality(self, df, threshold=0.05):
        """Check normality of numerical features and identify those needing transformation"""
        logging.info("\n" + "="*50)
        logging.info("NORMALITY TESTING")
        logging.info("="*50)
        
        non_normal_features = []
        
        for col in self.numerical_cols:
            stat, p_value = normaltest(df[col].dropna())
            skewness = skew(df[col].dropna())
            
            if p_value < threshold:
                non_normal_features.append(col)
                if abs(skewness) > 1:
                    if df[col].min() > 0:
                        self.log_transform_cols.append(col)
                        logging.info(f"  {col}: Non-normal (p={p_value:.4f}), skewness={skewness:.2f} - Will apply log transform")
                    else:
                        logging.info(f"  {col}: Non-normal (p={p_value:.4f}), skewness={skewness:.2f} - Contains non-positive values")
                else:
                    logging.info(f"  {col}: Non-normal (p={p_value:.4f}), skewness={skewness:.2f}")
            else:
                logging.info(f"  {col}: Normal (p={p_value:.4f})")
        
        return non_normal_features
    
    def apply_transformations(self, df):
        """Apply necessary transformations to features"""
        df_transformed = df.copy()
        
        for col in self.log_transform_cols:
            if col in df_transformed.columns:
                df_transformed[f'{col}_log'] = np.log1p(df_transformed[col])
        
        return df_transformed
    
    def encode_categorical(self, df, fit=True):
        """Encode categorical variables using label encoding and one-hot encoding."""
        df_encoded = df.copy()
        
        for col in self.label_encode_cols:
            if col in df_encoded.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
                else:
                    df_encoded[col] = self.label_encoders[col].transform(df_encoded[col].astype(str))

        if self.one_hot_encode_cols:
            df_to_encode = df_encoded[self.one_hot_encode_cols].astype(str)
            df_one_hot = pd.get_dummies(df_to_encode, prefix=self.one_hot_encode_cols, dummy_na=False)

            if fit:
                self.one_hot_encoded_columns = df_one_hot.columns.tolist()

            df_encoded = pd.concat([df_encoded.drop(columns=self.one_hot_encode_cols), df_one_hot], axis=1)

            if not fit:
                missing_cols = set(self.one_hot_encoded_columns) - set(df_encoded.columns)
                for c in missing_cols:
                    df_encoded[c] = 0
                
                df_encoded = df_encoded[list(df_encoded.columns.drop(self.one_hot_encoded_columns, errors='ignore')) + self.one_hot_encoded_columns]

        return df_encoded
    
    def select_features(self, X, y, n_features=15, method='mutual_info'):
        """Select best features using multiple methods"""
        logging.info("\n" + "="*50)
        logging.info("FEATURE SELECTION")
        logging.info("="*50)
        
        mi_selector = SelectKBest(score_func=mutual_info_classif, k=min(n_features, X.shape[1]))
        mi_selector.fit(X, y)
        mi_scores = pd.DataFrame({
            'feature': X.columns,
            'mi_score': mi_selector.scores_
        }).sort_values('mi_score', ascending=False)
        
        f_selector = SelectKBest(score_func=f_classif, k=min(n_features, X.shape[1]))
        f_selector.fit(X, y)
        f_scores = pd.DataFrame({
            'feature': X.columns,
            'f_score': f_selector.scores_
        }).sort_values('f_score', ascending=False)
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        rf_scores = pd.DataFrame({
            'feature': X.columns,
            'rf_importance': rf.feature_importances_
        }).sort_values('rf_importance', ascending=False)
        
        feature_ranks = pd.DataFrame({'feature': X.columns})
        feature_ranks['mi_rank'] = mi_scores.reset_index(drop=True).index + 1
        feature_ranks['f_rank'] = f_scores.reset_index(drop=True).index + 1
        feature_ranks['rf_rank'] = rf_scores.reset_index(drop=True).index + 1
        feature_ranks['avg_rank'] = feature_ranks[['mi_rank', 'f_rank', 'rf_rank']].mean(axis=1)
        feature_ranks = feature_ranks.sort_values('avg_rank')
        
        self.feature_importance_scores = {
            'mutual_info': mi_scores,
            'f_statistic': f_scores,
            'random_forest': rf_scores,
            'combined': feature_ranks
        }
        
        selected_features = feature_ranks.head(n_features)['feature'].tolist()
        logging.info(f"\nTop {len(selected_features)} features selected:")
        for i, feat in enumerate(selected_features, 1):
            avg_rank = feature_ranks[feature_ranks['feature'] == feat]['avg_rank'].values[0]
            logging.info(f"  {i}. {feat} (avg rank: {avg_rank:.2f})")
        
        return selected_features
    
    def preprocess(self, df, fit=True, select_features=True, n_features=15):
        """Complete preprocessing pipeline"""
        logging.info("\n" + "="*50)
        logging.info("PREPROCESSING PIPELINE")
        logging.info("="*50)
        
        if fit:
            self.identify_column_types(df)
        
        if fit:
            self.check_normality(df)
        df_transformed = self.apply_transformations(df)
        
        df_encoded = self.encode_categorical(df_transformed, fit=fit)
        
        X = df_encoded.drop(columns=self.id_cols + [self.target_col], errors='ignore')
        y = df_encoded[self.target_col] if self.target_col in df_encoded.columns else None
        
        if fit and select_features and y is not None:
            X = X.loc[:, X.var() > 0]
            selected_features = self.select_features(X, y, n_features=n_features)
            self.selected_features = selected_features
            X = X[selected_features]
        elif hasattr(self, 'selected_features'):
            for col in self.selected_features:
                if col not in X.columns:
                    X[col] = 0
            X = X[self.selected_features]

        if fit:
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
        
        return X_scaled, y


class HEDISModelTrainer:
    """
    Model trainer for HEDIS compliance prediction.
    Handles class imbalance, hyperparameter tuning, and model evaluation.
    """
    
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.models = {}
        self.best_params = {}
        self.cv_scores = {}
        self.predictions = {}
        self.use_smote = False
        
    def check_class_imbalance(self, y, threshold=0.3):
        """Check for class imbalance in target variable"""
        logging.info("\n" + "="*50)
        logging.info("CLASS IMBALANCE CHECK")
        logging.info("="*50)
        
        class_counts = pd.Series(y).value_counts()
        class_proportions = class_counts / len(y)
        
        logging.info("\nClass Distribution:")
        for cls, count in class_counts.items():
            logging.info(f"  Class {cls}: {count} samples ({class_proportions[cls]:.2%})")
        
        minority_proportion = class_proportions.min()
        if minority_proportion < threshold:
            logging.warning(f"\n⚠️  Class imbalance detected! Minority class: {minority_proportion:.2%}")
            logging.info("→ SMOTE will be applied during model training")
            self.use_smote = True
        else:
            logging.info(f"\n✓ Classes are relatively balanced")
            self.use_smote = False
        
        return class_counts
    
    def get_model_params(self):
        """Define hyperparameter search spaces for models"""
        return {
            'logistic': {
                'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
                'model__penalty': ['l1', 'l2'],
                'model__solver': ['liblinear', 'saga'],
                'model__max_iter': [500, 1000]
            },
            'xgboost': {
                'model__n_estimators': [50, 100, 200, 300],
                'model__max_depth': [3, 5, 7, 10],
                'model__learning_rate': [0.01, 0.05, 0.1, 0.3],
                'model__subsample': [0.6, 0.8, 1.0],
                'model__colsample_bytree': [0.6, 0.8, 1.0],
                'model__min_child_weight': [1, 3, 5],
                'model__gamma': [0, 0.1, 0.3]
            }
        }
    
    def train_model(self, X_train, y_train, model_type='logistic'):
        """Train a model with hyperparameter tuning"""
        logging.info(f"\n" + "="*50)
        logging.info(f"TRAINING {model_type.upper()} MODEL")
        logging.info("="*50)
        
        if model_type == 'logistic':
            base_model = LogisticRegression(random_state=42)
        elif model_type == 'xgboost':
            base_model = XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        if self.use_smote:
            pipeline = ImbPipeline([
                ('smote', SMOTE(random_state=42)),
                ('model', base_model)
            ])
        else:
            pipeline = ImbPipeline([
                ('model', base_model)
            ])
        
        param_dist = self.get_model_params()[model_type]
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        random_search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist,
            n_iter=20,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            random_state=42,
            verbose=1 # Note: This output from scikit-learn will go to console
        )
        
        logging.info(f"Performing randomized search with {cv.n_splits}-fold cross-validation...")
        random_search.fit(X_train, y_train)
        
        self.models[model_type] = random_search.best_estimator_
        self.best_params[model_type] = random_search.best_params_
        self.cv_scores[model_type] = {
            'mean': random_search.best_score_,
            'std': random_search.cv_results_['std_test_score'][random_search.best_index_]
        }
        
        logging.info(f"\nBest parameters:")
        for param, value in random_search.best_params_.items():
            logging.info(f"  {param}: {value}")
        logging.info(f"\nBest CV ROC-AUC: {random_search.best_score_:.4f} (±{self.cv_scores[model_type]['std']:.4f})")
        
        return random_search.best_estimator_
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        logging.info(f"\n{model_name.upper()} Performance:")
        logging.info(f"  Accuracy: {accuracy:.4f}")
        logging.info(f"  ROC-AUC: {roc_auc:.4f}")
        
        self.predictions[model_name] = {
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        return roc_auc


class HEDISVisualizer:
    """Visualization utilities for HEDIS compliance prediction"""
    
    @staticmethod
    def plot_class_distribution(y, title="Target Variable Distribution"):
        """Plot class distribution as a pie chart"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        class_counts = pd.Series(y).value_counts().sort_index()
        colors = ['#2E86AB', '#A23B72']
        
        labels = [f'Class {int(idx)}\n({count:,} members)' 
                 for idx, count in zip(class_counts.index, class_counts.values)]
        
        wedges, texts, autotexts = ax.pie(
            class_counts.values,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            explode=(0.05, 0.05),
            shadow=True,
            textprops={'fontsize': 12, 'fontweight': 'bold'},
            wedgeprops={'edgecolor': 'black', 'linewidth': 2}
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(14)
            autotext.set_fontweight('bold')
        
        for text in texts:
            text.set_fontsize(12)
            text.set_fontweight('bold')
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        legend_labels = [
            f'Non-compliant (0): {class_counts.iloc[0]:,} ({class_counts.iloc[0]/len(y):.1%})',
            f'Compliant (1): {class_counts.iloc[1]:,} ({class_counts.iloc[1]/len(y):.1%})'
        ]
        ax.legend(wedges, legend_labels, title="HEDIS Compliance Status", 
                 loc="upper right", bbox_to_anchor=(1.25, 1), fontsize=11,
                 title_fontsize=12, frameon=True, fancybox=True, shadow=True)
        
        ax.axis('equal')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_feature_importance(preprocessor, top_n=15):
        """Plot feature importance from multiple methods"""
        if not preprocessor.feature_importance_scores:
            logging.warning("No feature importance scores available to plot.")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.subplots_adjust(hspace=0.4, wspace=0.5)

        # Mutual Information
        mi_scores = preprocessor.feature_importance_scores['mutual_info'].head(top_n).sort_values('mi_score')
        axes[0, 0].barh(mi_scores['feature'], mi_scores['mi_score'], color='#2E86AB')
        axes[0, 0].set_xlabel('Mutual Information Score', fontweight='bold')
        axes[0, 0].set_title('Feature Importance - Mutual Information', fontweight='bold')
        axes[0, 0].grid(axis='x', alpha=0.3)
        
        # F-statistic
        f_scores = preprocessor.feature_importance_scores['f_statistic'].head(top_n).sort_values('f_score')
        axes[0, 1].barh(f_scores['feature'], f_scores['f_score'], color='#A23B72')
        axes[0, 1].set_xlabel('F-statistic Score', fontweight='bold')
        axes[0, 1].set_title('Feature Importance - ANOVA F-statistic', fontweight='bold')
        axes[0, 1].grid(axis='x', alpha=0.3)
        
        # Random Forest
        rf_scores = preprocessor.feature_importance_scores['random_forest'].head(top_n).sort_values('rf_importance')
        axes[1, 0].barh(rf_scores['feature'], rf_scores['rf_importance'], color='#F18F01')
        axes[1, 0].set_xlabel('Random Forest Importance', fontweight='bold')
        axes[1, 0].set_title('Feature Importance - Random Forest', fontweight='bold')
        axes[1, 0].grid(axis='x', alpha=0.3)
        
        # Combined Ranking
        combined = preprocessor.feature_importance_scores['combined'].head(top_n).sort_values('avg_rank', ascending=False)
        axes[1, 1].barh(combined['feature'], 1/combined['avg_rank'], color='#C73E1D')
        axes[1, 1].set_xlabel('Combined Score (1/avg_rank)', fontweight='bold')
        axes[1, 1].set_title('Feature Importance - Combined Ranking', fontweight='bold')
        axes[1, 1].grid(axis='x', alpha=0.3)
        
        plt.suptitle('Feature Selection Results - Multiple Methods', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_roc_curves(trainer, X_test, y_test):
        """Plot ROC curves for all models"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        colors = {'logistic': '#2E86AB', 'xgboost': '#A23B72'}
        
        for model_name, predictions in trainer.predictions.items():
            fpr, tpr, _ = roc_curve(y_test, predictions['y_pred_proba'])
            auc = predictions['roc_auc']
            
            ax.plot(fpr, tpr, color=colors.get(model_name, 'black'), linewidth=2.5,
                   label=f'{model_name.upper()} (AUC = {auc:.4f})')
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=1.5)
        
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_classification_reports(trainer):
        """Plot classification reports as heatmaps"""
        n_models = len(trainer.predictions)
        fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6))
        axes = np.atleast_1d(axes)
        
        for idx, (model_name, predictions) in enumerate(trainer.predictions.items()):
            report = predictions['classification_report']
            
            df_report = pd.DataFrame(report).transpose()
            df_report = df_report.iloc[:-3, :-1]
            
            sns.heatmap(df_report, annot=True, fmt='.3f', cmap='YlOrRd', 
                       ax=axes[idx], cbar_kws={'label': 'Score'},
                       vmin=0, vmax=1, linewidths=1, linecolor='black')
            axes[idx].set_title(f'{model_name.upper()} - Classification Report', 
                               fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('Classes', fontweight='bold')
            axes[idx].set_xlabel('Metrics', fontweight='bold')
        
        plt.suptitle('Model Performance Metrics', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_confusion_matrices(trainer, X_test, y_test):
        """Plot confusion matrices"""
        n_models = len(trainer.predictions)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
        axes = np.atleast_1d(axes)

        for idx, (model_name, predictions) in enumerate(trainer.predictions.items()):
            cm = confusion_matrix(y_test, predictions['y_pred'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       cbar_kws={'label': 'Count'}, linewidths=1, linecolor='black')
            axes[idx].set_title(f'{model_name.upper()} - Confusion Matrix', 
                               fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('True Label', fontweight='bold')
            axes[idx].set_xlabel('Predicted Label', fontweight='bold')
            
            accuracy = predictions['accuracy']
            axes[idx].text(0.5, -0.15, f'Accuracy: {accuracy:.4f}', 
                          transform=axes[idx].transAxes, ha='center', 
                          fontsize=11, fontweight='bold')
        
        plt.suptitle('Confusion Matrices', fontsize=14, fontweight='bold', y=1.05)
        plt.tight_layout()
        return fig


def main():
    """Main execution function"""
    # Setup logging to file and console
    setup_logging(log_dir="output")

    logging.info("="*60)
    logging.info("HEDIS COMPLIANCE PREDICTION PIPELINE")
    logging.info("="*60)
    
    # Generate sample data
    np.random.seed(42)
    n = 10000
    data = {
        "src_mbr_id": np.arange(1, n + 1),
        "age": np.random.randint(0, 100, n),
        "sex": np.random.choice(["M", "F"], size=n),
        "racetype": np.random.choice(["01", "02", "03", "04", "05", "06"], size=n),
        "ethnicitytype": np.random.choice(["11", "12", "13", "14", "15", '16'], size=n),
        "numercnt": np.random.choice([0, 1], size=n, p=[0.85, 0.15]),
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
        "social_risk_score": np.random.uniform(0, 100, n),
        "multiple_measures": np.random.randint(1, 4, n),
        "num_sum": np.random.randint(0, 3, n),
        "compliance_ratio": np.random.uniform(0, 1, n),
    }
    df = pd.DataFrame(data)
    
    id_columns = ["src_mbr_id", "racetype", "ethnicitytype"]
    for col in id_columns:
        df[col] = df[col].astype(str)
    
    round_columns = [c for c in df.select_dtypes(include=np.number).columns if c not in ['src_mbr_id', 'numercnt', 'age']]
    for col in round_columns:
        df[col] = round(df[col].astype(float), 2)
    
    logging.info(f"\nDataset shape: {df.shape}")
    logging.info(f"Target variable: numercnt")
    
    preprocessor = HEDISPreprocessor(target_col='numercnt', id_cols=['src_mbr_id'])
    
    X, y = preprocessor.preprocess(df, fit=True, select_features=True, n_features=20)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logging.info(f"\nTraining set size: {X_train.shape}")
    logging.info(f"Test set size: {X_test.shape}")
    
    trainer = HEDISModelTrainer(preprocessor)
    trainer.check_class_imbalance(y_train)
    
    logistic_model = trainer.train_model(X_train, y_train, model_type='logistic')
    xgboost_model = trainer.train_model(X_train, y_train, model_type='xgboost')
    
    logging.info("\n" + "="*50)
    logging.info("MODEL EVALUATION ON TEST SET")
    logging.info("="*50)
    
    trainer.evaluate_model(logistic_model, X_test, y_test, 'logistic')
    trainer.evaluate_model(xgboost_model, X_test, y_test, 'xgboost')
    
    output_dir = "output/plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logging.info(f"\n📊 Saving charts to '{output_dir}' directory...")
    
    visualizer = HEDISVisualizer()
    
    plots = {
        "01_class_distribution": visualizer.plot_class_distribution(y, title="HEDIS Compliance Distribution (numercnt)"),
        "02_feature_importance": visualizer.plot_feature_importance(preprocessor, top_n=15),
        "03_roc_curves": visualizer.plot_roc_curves(trainer, X_test, y_test),
        "04_classification_reports": visualizer.plot_classification_reports(trainer),
        "05_confusion_matrices": visualizer.plot_confusion_matrices(trainer, X_test, y_test)
    }

    for name, fig in plots.items():
        if fig:
            try:
                fig.savefig(f"{output_dir}/{name}.png", dpi=300, bbox_inches='tight')
                plt.close(fig)
                logging.info(f"  ✓ Saved {name.replace('_', ' ')} chart")
            except Exception as e:
                logging.error(f"  ✗ Failed to save {name}.png. Error: {e}")


    logging.info(f"\n✅ All charts saved successfully to '{output_dir}' directory!")

    logging.info("\n" + "="*50)
    logging.info("MODEL COMPARISON SUMMARY")
    logging.info("="*50)
    
    comparison_df = pd.DataFrame({
        'Model': ['Logistic Regression', 'XGBoost'],
        'CV ROC-AUC': [
            f"{trainer.cv_scores['logistic']['mean']:.4f} (±{trainer.cv_scores['logistic']['std']:.4f})",
            f"{trainer.cv_scores['xgboost']['mean']:.4f} (±{trainer.cv_scores['xgboost']['std']:.4f})"
        ],
        'Test ROC-AUC': [
            trainer.predictions['logistic']['roc_auc'],
            trainer.predictions['xgboost']['roc_auc']
        ],
        'Test Accuracy': [
            trainer.predictions['logistic']['accuracy'],
            trainer.predictions['xgboost']['accuracy']
        ]
    })
    
    logging.info(f"\n{comparison_df.to_string(index=False)}")
    
    best_model = 'xgboost' if trainer.predictions['xgboost']['roc_auc'] > trainer.predictions['logistic']['roc_auc'] else 'logistic'
    logging.info(f"\n🏆 Best performing model: {best_model.upper()} (ROC-AUC: {trainer.predictions[best_model]['roc_auc']:.4f})")
    
    return preprocessor, trainer


if __name__ == "__main__":
    preprocessor, trainer = main()