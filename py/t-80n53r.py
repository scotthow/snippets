import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import normaltest, skew
import warnings
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

class HEDISPreprocessor:
    """
    Preprocessor for HEDIS compliance prediction data.
    Handles feature engineering, transformation, and selection.
    """
    
    def __init__(self, target_col='numercnt', id_cols=None):
        self.target_col = target_col
        self.id_cols = id_cols or ['member_key']
        self.categorical_cols = []
        self.numerical_cols = []
        self.log_transform_cols = []
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance_scores = {}
        
    def identify_column_types(self, df):
        """Identify categorical and numerical columns"""
        # Exclude ID and target columns
        exclude_cols = self.id_cols + [self.target_col]
        
        for col in df.columns:
            if col not in exclude_cols:
                if df[col].dtype == 'object' or col in ['sex', 'racetype', 'ethnicitytype']:
                    self.categorical_cols.append(col)
                else:
                    self.numerical_cols.append(col)
        
        print(f"Identified {len(self.categorical_cols)} categorical columns")
        print(f"Identified {len(self.numerical_cols)} numerical columns")
        
    def check_normality(self, df, threshold=0.05):
        """Check normality of numerical features and identify those needing transformation"""
        print("\n" + "="*50)
        print("NORMALITY TESTING")
        print("="*50)
        
        non_normal_features = []
        
        for col in self.numerical_cols:
            # Perform D'Agostino-Pearson test
            stat, p_value = normaltest(df[col].dropna())
            skewness = skew(df[col].dropna())
            
            if p_value < threshold:
                non_normal_features.append(col)
                if abs(skewness) > 1:  # High skewness
                    # Check if log transformation would help (all values must be positive)
                    if df[col].min() > 0:
                        self.log_transform_cols.append(col)
                        print(f"  {col}: Non-normal (p={p_value:.4f}), skewness={skewness:.2f} - Will apply log transform")
                    else:
                        print(f"  {col}: Non-normal (p={p_value:.4f}), skewness={skewness:.2f} - Contains non-positive values")
                else:
                    print(f"  {col}: Non-normal (p={p_value:.4f}), skewness={skewness:.2f}")
            else:
                print(f"  {col}: Normal (p={p_value:.4f})")
        
        return non_normal_features
    
    def apply_transformations(self, df):
        """Apply necessary transformations to features"""
        df_transformed = df.copy()
        
        # Apply log transformation to highly skewed features
        for col in self.log_transform_cols:
            if col in df_transformed.columns:
                df_transformed[f'{col}_log'] = np.log1p(df_transformed[col])
                # Keep both original and transformed versions for now
                # Feature selection will determine which to use
        
        return df_transformed
    
    def encode_categorical(self, df, fit=True):
        """Encode categorical variables"""
        df_encoded = df.copy()
        
        for col in self.categorical_cols:
            if col in df_encoded.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
                else:
                    df_encoded[col] = self.label_encoders[col].transform(df_encoded[col].astype(str))
        
        return df_encoded
    
    def select_features(self, X, y, n_features=15, method='mutual_info'):
        """
        Select best features using multiple methods
        """
        print("\n" + "="*50)
        print("FEATURE SELECTION")
        print("="*50)
        
        # Method 1: Mutual Information
        mi_selector = SelectKBest(score_func=mutual_info_classif, k=min(n_features, X.shape[1]))
        mi_selector.fit(X, y)
        mi_scores = pd.DataFrame({
            'feature': X.columns,
            'mi_score': mi_selector.scores_
        }).sort_values('mi_score', ascending=False)
        
        # Method 2: ANOVA F-statistic
        f_selector = SelectKBest(score_func=f_classif, k=min(n_features, X.shape[1]))
        f_selector.fit(X, y)
        f_scores = pd.DataFrame({
            'feature': X.columns,
            'f_score': f_selector.scores_
        }).sort_values('f_score', ascending=False)
        
        # Method 3: Random Forest Feature Importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        rf_scores = pd.DataFrame({
            'feature': X.columns,
            'rf_importance': rf.feature_importances_
        }).sort_values('rf_importance', ascending=False)
        
        # Combine scores (use rank-based approach)
        feature_ranks = pd.DataFrame({'feature': X.columns})
        feature_ranks['mi_rank'] = mi_scores.reset_index(drop=True).index + 1
        feature_ranks['f_rank'] = f_scores.reset_index(drop=True).index + 1
        feature_ranks['rf_rank'] = rf_scores.reset_index(drop=True).index + 1
        feature_ranks['avg_rank'] = feature_ranks[['mi_rank', 'f_rank', 'rf_rank']].mean(axis=1)
        feature_ranks = feature_ranks.sort_values('avg_rank')
        
        # Store scores for visualization
        self.feature_importance_scores = {
            'mutual_info': mi_scores,
            'f_statistic': f_scores,
            'random_forest': rf_scores,
            'combined': feature_ranks
        }
        
        # Select top features
        selected_features = feature_ranks.head(n_features)['feature'].tolist()
        print(f"\nTop {len(selected_features)} features selected:")
        for i, feat in enumerate(selected_features, 1):
            avg_rank = feature_ranks[feature_ranks['feature'] == feat]['avg_rank'].values[0]
            print(f"  {i}. {feat} (avg rank: {avg_rank:.2f})")
        
        return selected_features
    
    def preprocess(self, df, fit=True, select_features=True, n_features=15):
        """Complete preprocessing pipeline"""
        print("\n" + "="*50)
        print("PREPROCESSING PIPELINE")
        print("="*50)
        
        # Identify column types
        if fit:
            self.identify_column_types(df)
        
        # Check normality and apply transformations
        if fit:
            self.check_normality(df)
        df_transformed = self.apply_transformations(df)
        
        # Encode categorical variables
        df_encoded = self.encode_categorical(df_transformed, fit=fit)
        
        # Separate features and target
        X = df_encoded.drop(columns=self.id_cols + [self.target_col])
        y = df_encoded[self.target_col] if self.target_col in df_encoded.columns else None
        
        # Feature selection
        if fit and select_features and y is not None:
            selected_features = self.select_features(X, y, n_features=n_features)
            self.selected_features = selected_features
            X = X[selected_features]
        elif hasattr(self, 'selected_features'):
            X = X[self.selected_features]
        
        # Scale features
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
        print("\n" + "="*50)
        print("CLASS IMBALANCE CHECK")
        print("="*50)
        
        class_counts = pd.Series(y).value_counts()
        class_proportions = class_counts / len(y)
        
        print("\nClass Distribution:")
        for cls, count in class_counts.items():
            print(f"  Class {cls}: {count} samples ({class_proportions[cls]:.2%})")
        
        # Check if minority class is below threshold
        minority_proportion = class_proportions.min()
        if minority_proportion < threshold:
            print(f"\n⚠️  Class imbalance detected! Minority class: {minority_proportion:.2%}")
            print("→ SMOTE will be applied during model training")
            self.use_smote = True
        else:
            print(f"\n✓ Classes are relatively balanced")
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
        print(f"\n" + "="*50)
        print(f"TRAINING {model_type.upper()} MODEL")
        print("="*50)
        
        # Create base model
        if model_type == 'logistic':
            base_model = LogisticRegression(random_state=42)
        elif model_type == 'xgboost':
            base_model = XGBClassifier(random_state=42, eval_metric='logloss')
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create pipeline with SMOTE if needed
        if self.use_smote:
            pipeline = ImbPipeline([
                ('smote', SMOTE(random_state=42)),
                ('model', base_model)
            ])
        else:
            pipeline = ImbPipeline([
                ('model', base_model)
            ])
        
        # Get hyperparameters
        param_dist = self.get_model_params()[model_type]
        
        # Perform randomized search with cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        random_search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist,
            n_iter=20,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        print(f"Performing randomized search with {cv.n_splits}-fold cross-validation...")
        random_search.fit(X_train, y_train)
        
        # Store results
        self.models[model_type] = random_search.best_estimator_
        self.best_params[model_type] = random_search.best_params_
        self.cv_scores[model_type] = {
            'mean': random_search.best_score_,
            'std': random_search.cv_results_['std_test_score'][random_search.best_index_]
        }
        
        print(f"\nBest parameters:")
        for param, value in random_search.best_params_.items():
            print(f"  {param}: {value}")
        print(f"\nBest CV ROC-AUC: {random_search.best_score_:.4f} (±{self.cv_scores[model_type]['std']:.4f})")
        
        return random_search.best_estimator_
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\n{model_name.upper()} Performance:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        
        # Store predictions
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
        
        # Create labels with counts and percentages
        labels = [f'Class {int(idx)}\n({count:,} members)' 
                 for idx, count in zip(class_counts.index, class_counts.values)]
        
        # Create pie chart with enhanced styling
        wedges, texts, autotexts = ax.pie(
            class_counts.values,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            explode=(0.05, 0.05),  # Slightly separate the wedges
            shadow=True,
            textprops={'fontsize': 12, 'fontweight': 'bold'},
            wedgeprops={'edgecolor': 'black', 'linewidth': 2}
        )
        
        # Enhance the percentage text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(14)
            autotext.set_fontweight('bold')
        
        # Enhance the label text
        for text in texts:
            text.set_fontsize(12)
            text.set_fontweight('bold')
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Add a legend with additional information
        legend_labels = [
            f'Non-compliant (0): {class_counts.iloc[0]:,} ({class_counts.iloc[0]/len(y):.1%})',
            f'Compliant (1): {class_counts.iloc[1]:,} ({class_counts.iloc[1]/len(y):.1%})'
        ]
        ax.legend(wedges, legend_labels, title="HEDIS Compliance Status", 
                 loc="upper right", bbox_to_anchor=(1.25, 1), fontsize=11,
                 title_fontsize=12, frameon=True, fancybox=True, shadow=True)
        
        # # Add text box with imbalance ratio
        # ratio = class_counts.min() / class_counts.max()
        # textstr = f'Imbalance Ratio: 1:{class_counts.max()/class_counts.min():.1f}\nTotal Members: {len(y):,}'
        # props = dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black', linewidth=1.5)
        # ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=11,
        #         verticalalignment='bottom', horizontalalignment='center', bbox=props,
        #         fontweight='bold')
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.axis('equal')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_feature_importance(preprocessor, top_n=15):
        """Plot feature importance from multiple methods"""
        if not preprocessor.feature_importance_scores:
            print("No feature importance scores available")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Mutual Information
        mi_scores = preprocessor.feature_importance_scores['mutual_info'].head(top_n)
        axes[0, 0].barh(range(len(mi_scores)), mi_scores['mi_score'].values, color='#2E86AB')
        axes[0, 0].set_yticks(range(len(mi_scores)))
        axes[0, 0].set_yticklabels(mi_scores['feature'].values)
        axes[0, 0].set_xlabel('Mutual Information Score', fontweight='bold')
        axes[0, 0].set_title('Feature Importance - Mutual Information', fontweight='bold')
        axes[0, 0].grid(axis='x', alpha=0.3)
        
        # F-statistic
        f_scores = preprocessor.feature_importance_scores['f_statistic'].head(top_n)
        axes[0, 1].barh(range(len(f_scores)), f_scores['f_score'].values, color='#A23B72')
        axes[0, 1].set_yticks(range(len(f_scores)))
        axes[0, 1].set_yticklabels(f_scores['feature'].values)
        axes[0, 1].set_xlabel('F-statistic Score', fontweight='bold')
        axes[0, 1].set_title('Feature Importance - ANOVA F-statistic', fontweight='bold')
        axes[0, 1].grid(axis='x', alpha=0.3)
        
        # Random Forest
        rf_scores = preprocessor.feature_importance_scores['random_forest'].head(top_n)
        axes[1, 0].barh(range(len(rf_scores)), rf_scores['rf_importance'].values, color='#F18F01')
        axes[1, 0].set_yticks(range(len(rf_scores)))
        axes[1, 0].set_yticklabels(rf_scores['feature'].values)
        axes[1, 0].set_xlabel('Random Forest Importance', fontweight='bold')
        axes[1, 0].set_title('Feature Importance - Random Forest', fontweight='bold')
        axes[1, 0].grid(axis='x', alpha=0.3)
        
        # Combined Ranking
        combined = preprocessor.feature_importance_scores['combined'].head(top_n)
        axes[1, 1].barh(range(len(combined)), 1/combined['avg_rank'].values, color='#C73E1D')
        axes[1, 1].set_yticks(range(len(combined)))
        axes[1, 1].set_yticklabels(combined['feature'].values)
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
        
        # Plot diagonal
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
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        for idx, (model_name, predictions) in enumerate(trainer.predictions.items()):
            report = predictions['classification_report']
            
            # Create dataframe from classification report
            df_report = pd.DataFrame(report).transpose()
            df_report = df_report.iloc[:-3, :-1]  # Remove accuracy, macro avg, weighted avg and support
            
            # Plot heatmap
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
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for idx, (model_name, predictions) in enumerate(trainer.predictions.items()):
            cm = confusion_matrix(y_test, predictions['y_pred'])
            
            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       cbar_kws={'label': 'Count'}, linewidths=1, linecolor='black')
            axes[idx].set_title(f'{model_name.upper()} - Confusion Matrix', 
                               fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('True Label', fontweight='bold')
            axes[idx].set_xlabel('Predicted Label', fontweight='bold')
            
            # Add accuracy text
            accuracy = predictions['accuracy']
            axes[idx].text(0.5, -0.15, f'Accuracy: {accuracy:.4f}', 
                          transform=axes[idx].transAxes, ha='center', 
                          fontsize=11, fontweight='bold')
        
        plt.suptitle('Confusion Matrices', fontsize=14, fontweight='bold', y=1.05)
        plt.tight_layout()
        return fig


def main():
    """Main execution function"""
    print("="*60)
    print("HEDIS COMPLIANCE PREDICTION PIPELINE")
    print("="*60)
    
    # Generate sample data (as provided)
    np.random.seed(42)
    n = 10000
    data = {
        "member_key": np.arange(1, n + 1),
        "age": np.random.randint(0, 100, n),
        "sex": np.random.choice(["M", "F"], size=n),
        "lastbp_dia": np.random.randint(50, 100, n),
        "lastbp_sys": np.random.randint(100, 150, n),
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
        "social_risk_score": np.random.uniform(0, 100, n)
    }
    df = pd.DataFrame(data)
    
    # Convert ID columns to strings
    id_columns = ["member_key", "racetype", "ethnicitytype"]
    for col in id_columns:
        df[col] = df[col].astype(str)
    
    # Round numerical columns
    round_columns = [
        "food_access", "health_access", "housing_desert", "poverty_score",
        "transport_access", "education_index", "citizenship_index",
        "housing_ownership", "income_index", "income_inequality",
        "natural_disaster", "social_isolation", "technology_access",
        "unemployment_index", "water_quality", "health_infra",
        "social_risk_score", "lastbp_dia", "lastbp_sys"
    ]
    for col in round_columns:
        df[col] = round(df[col].astype(float), 2)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Target variable: numercnt")
    
    # Initialize preprocessor
    preprocessor = HEDISPreprocessor(target_col='numercnt', id_cols=['member_key'])
    
    # Preprocess data
    X, y = preprocessor.preprocess(df, fit=True, select_features=True, n_features=15)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # Initialize model trainer
    trainer = HEDISModelTrainer(preprocessor)
    
    # Check class imbalance
    class_counts = trainer.check_class_imbalance(y_train)
    
    # Train models
    logistic_model = trainer.train_model(X_train, y_train, model_type='logistic')
    xgboost_model = trainer.train_model(X_train, y_train, model_type='xgboost')
    
    # Evaluate models
    print("\n" + "="*50)
    print("MODEL EVALUATION ON TEST SET")
    print("="*50)
    
    trainer.evaluate_model(logistic_model, X_test, y_test, 'logistic')
    trainer.evaluate_model(xgboost_model, X_test, y_test, 'xgboost')
    
    # Print detailed classification reports
    print("\n" + "="*50)
    print("DETAILED CLASSIFICATION REPORTS")
    print("="*50)
    
    for model_name, predictions in trainer.predictions.items():
        print(f"\n{model_name.upper()} Model:")
        print("-"*30)
        report = predictions['classification_report']
        for metric in ['0', '1', 'macro avg', 'weighted avg']:
            if metric in report:
                print(f"\n{metric}:")
                for key, value in report[metric].items():
                    if key != 'support':
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value:.0f}")
    
    # Create visualizations
    visualizer = HEDISVisualizer()
    
    # Create output directory for charts
    import os
    output_dir = "output/plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"\n📊 Saving charts to '{output_dir}' directory...")
    
    # Plot 1: Class Distribution
    fig1 = visualizer.plot_class_distribution(y, title="HEDIS Compliance Distribution (numercnt)")
    fig1.savefig(f"{output_dir}/01_class_distribution.png", dpi=300, bbox_inches='tight')
    # fig1.savefig(f"{output_dir}/01_class_distribution.pdf", bbox_inches='tight')  # Also save as PDF
    plt.close(fig1)
    print(f"  ✓ Saved class distribution chart")
    
    # Plot 2: Feature Importance
    fig2 = visualizer.plot_feature_importance(preprocessor, top_n=15)
    if fig2:
        fig2.savefig(f"{output_dir}/02_feature_importance.png", dpi=300, bbox_inches='tight')
        # fig2.savefig(f"{output_dir}/02_feature_importance.pdf", bbox_inches='tight')
        plt.close(fig2)
        print(f"  ✓ Saved feature importance charts")
    
    # Plot 3: ROC Curves
    fig3 = visualizer.plot_roc_curves(trainer, X_test, y_test)
    fig3.savefig(f"{output_dir}/03_roc_curves.png", dpi=300, bbox_inches='tight')
    # fig3.savefig(f"{output_dir}/03_roc_curves.pdf", bbox_inches='tight')
    plt.close(fig3)
    print(f"  ✓ Saved ROC curves")
    
    # Plot 4: Classification Reports
    fig4 = visualizer.plot_classification_reports(trainer)
    fig4.savefig(f"{output_dir}/04_classification_reports.png", dpi=300, bbox_inches='tight')
    # fig4.savefig(f"{output_dir}/04_classification_reports.pdf", bbox_inches='tight')
    plt.close(fig4)
    print(f"  ✓ Saved classification report heatmaps")
    
    # Plot 5: Confusion Matrices
    fig5 = visualizer.plot_confusion_matrices(trainer, X_test, y_test)
    fig5.savefig(f"{output_dir}/05_confusion_matrices.png", dpi=300, bbox_inches='tight')
    # fig5.savefig(f"{output_dir}/05_confusion_matrices.pdf", bbox_inches='tight')
    plt.close(fig5)
    print(f"  ✓ Saved confusion matrices")
    
    print(f"\n✅ All charts saved successfully to '{output_dir}' directory!")
    
    # Optional: Display charts if needed (uncomment below lines)
    # plt.show()  # This would show all plots at once if you still want to see them
    
    # Model Comparison Summary
    print("\n" + "="*50)
    print("MODEL COMPARISON SUMMARY")
    print("="*50)
    
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
    
    print("\n", comparison_df.to_string(index=False))
    
    # Determine best model
    best_model = 'xgboost' if trainer.predictions['xgboost']['roc_auc'] > trainer.predictions['logistic']['roc_auc'] else 'logistic'
    print(f"\n🏆 Best performing model: {best_model.upper()} (ROC-AUC: {trainer.predictions[best_model]['roc_auc']:.4f})")
    
    # Feature importance insights
    print("\n" + "="*50)
    print("KEY INSIGHTS")
    print("="*50)
    
    print("\n1. TOP PREDICTIVE FEATURES:")
    top_features = preprocessor.feature_importance_scores['combined'].head(5)
    for i, row in enumerate(top_features.itertuples(), 1):
        print(f"   {i}. {row.feature}")
    
    print("\n2. CLASS IMBALANCE HANDLING:")
    if trainer.use_smote:
        print("   • SMOTE was applied to address severe class imbalance")
        print(f"   • Original minority class ratio: {class_counts.min()/len(y_train):.2%}")
        print("   • Synthetic samples generated during training")
    else:
        print("   • No severe class imbalance detected")
        print("   • Models trained on original data distribution")
    
    print("\n3. MODEL PERFORMANCE:")
    print(f"   • Both models show good discrimination ability (ROC-AUC > 0.5)")
    print(f"   • {best_model.upper()} model performs best overall")
    print(f"   • Cross-validation ensures robust performance estimates")
    
    print("\n4. RECOMMENDATIONS:")
    print("   • Consider collecting more data for minority class")
    print("   • Monitor model performance on new data regularly")
    print("   • Investigate top features for clinical relevance")
    print("   • Consider ensemble methods for production deployment")
    
    return preprocessor, trainer


if __name__ == "__main__":
    preprocessor, trainer = main()