import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, precision_recall_curve, f1_score)
from sklearn.feature_selection import SelectKBest, f_classif, RFE, mutual_info_classif
from imblearn.over_sampling import SMOTE
from scipy import stats
from scipy.stats import normaltest, skew, kurtosis
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set style for professional charts
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class HEDISPreprocessor:
    """
    Comprehensive preprocessing pipeline for HEDIS compliance prediction
    """
    
    def __init__(self, target_col='numercnt', random_state=42):
        self.target_col = target_col
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.power_transformer = PowerTransformer(method='yeo-johnson')
        self.feature_names = None
        self.selected_features = None
        self.smote = SMOTE(random_state=random_state)
        self.normality_results = {}
        self.log_transformed_cols = []
        self.power_transformed_cols = []
        
    def check_class_imbalance(self, y):
        """Check and visualize class imbalance"""
        unique, counts = np.unique(y, return_counts=True)
        imbalance_ratio = counts.min() / counts.max()
        
        print("=" * 60)
        print("CLASS IMBALANCE ANALYSIS")
        print("=" * 60)
        print(f"Class distribution:")
        for cls, cnt in zip(unique, counts):
            print(f"  Class {cls}: {cnt} samples ({cnt/len(y)*100:.1f}%)")
        print(f"Imbalance ratio: {imbalance_ratio:.3f}")
        
        if imbalance_ratio < 0.4:
            print("⚠️  Significant class imbalance detected - SMOTE will be applied")
        else:
            print("✓ Class distribution is relatively balanced")
        print()
        
        return imbalance_ratio < 0.4
    
    def test_normality(self, X, feature_names):
        """Test features for normality and apply transformations"""
        print("=" * 60)
        print("NORMALITY TESTING & TRANSFORMATION")
        print("=" * 60)
        
        X_transformed = X.copy()
        
        for i, col in enumerate(feature_names):
            if col in ['racetype', 'ethnicitytype']:  # Skip categorical
                continue
                
            data = X[:, i]
            
            # Skip if constant or near-constant
            if np.std(data) < 1e-10:
                continue
            
            # Test original distribution
            stat, p_value = normaltest(data)
            skewness = skew(data)
            kurt = kurtosis(data)
            
            self.normality_results[col] = {
                'original_p': p_value,
                'original_skew': skewness,
                'original_kurtosis': kurt,
                'is_normal': p_value > 0.05
            }
            
            # Apply transformation if not normal and skewed
            if p_value < 0.05 and abs(skewness) > 0.5:
                # Try log transformation for positive skewed data
                if skewness > 0 and np.all(data > 0):
                    transformed = np.log1p(data)
                    trans_stat, trans_p = normaltest(transformed)
                    
                    if trans_p > p_value:  # Improvement
                        X_transformed[:, i] = transformed
                        self.log_transformed_cols.append(col)
                        self.normality_results[col]['transform'] = 'log'
                        self.normality_results[col]['transformed_p'] = trans_p
                        print(f"  {col}: Applied log transformation (p-value: {p_value:.4f} → {trans_p:.4f})")
                    else:
                        # Try power transformation
                        transformed = self.power_transformer.fit_transform(data.reshape(-1, 1)).ravel()
                        trans_stat, trans_p = normaltest(transformed)
                        if trans_p > p_value:
                            X_transformed[:, i] = transformed
                            self.power_transformed_cols.append(col)
                            self.normality_results[col]['transform'] = 'power'
                            self.normality_results[col]['transformed_p'] = trans_p
                            print(f"  {col}: Applied power transformation (p-value: {p_value:.4f} → {trans_p:.4f})")
        
        print(f"\nTransformations applied:")
        print(f"  Log transformation: {len(self.log_transformed_cols)} features")
        print(f"  Power transformation: {len(self.power_transformed_cols)} features")
        print()
        
        return X_transformed
    
    def engineer_features(self, df):
        """Create domain-specific engineered features"""
        df_eng = df.copy()
        
        # Social vulnerability composite scores
        df_eng['social_vulnerability'] = (
            df_eng['poverty_score'] * 0.3 +
            df_eng['social_isolation'] * 0.2 +
            df_eng['unemployment_index'] * 0.2 +
            (100 - df_eng['education_index']) * 0.15 +
            (100 - df_eng['income_index']) * 0.15
        )
        
        # Access composite score
        df_eng['access_composite'] = (
            df_eng['food_access'] * 0.25 +
            df_eng['health_access'] * 0.35 +
            df_eng['transport_access'] * 0.25 +
            df_eng['technology_access'] * 0.15
        )
        
        # Housing stability index
        df_eng['housing_stability'] = (
            df_eng['housing_ownership'] * 0.5 +
            (100 - df_eng['housing_desert']) * 0.5
        )
        
        # Environmental risk score
        df_eng['environmental_risk'] = (
            df_eng['natural_disaster'] * 0.6 +
            (1 - df_eng['water_quality']) * 100 * 0.4
        )
        
        # Age-based risk categories
        df_eng['age_group'] = pd.cut(df_eng['age'], 
                                     bins=[0, 18, 40, 65, 100],
                                     labels=[1, 2, 3, 4]).astype(int)
        
        # Interaction features
        df_eng['age_poverty_interaction'] = df_eng['age'] * df_eng['poverty_score'] / 100
        df_eng['health_social_interaction'] = df_eng['health_access'] * df_eng['social_risk_score'] / 100
        
        print("✓ Feature engineering completed - Added 7 composite features")
        
        return df_eng
    
    def select_features(self, X, y, feature_names, n_features=15):
        """Multi-method feature selection"""
        print("=" * 60)
        print("FEATURE SELECTION")
        print("=" * 60)
        
        # Method 1: Univariate F-test
        selector_f = SelectKBest(f_classif, k=min(n_features, X.shape[1]))
        selector_f.fit(X, y)
        f_scores = selector_f.scores_
        
        # Method 2: Mutual Information
        mi_scores = mutual_info_classif(X, y, random_state=self.random_state)
        
        # Method 3: Random Forest importance
        rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        rf.fit(X, y)
        rf_importance = rf.feature_importances_
        
        # Combine scores (normalized)
        f_scores_norm = f_scores / f_scores.max()
        mi_scores_norm = mi_scores / mi_scores.max()
        rf_scores_norm = rf_importance / rf_importance.max()
        
        combined_scores = (f_scores_norm + mi_scores_norm + rf_scores_norm) / 3
        
        # Select top features
        top_indices = np.argsort(combined_scores)[-n_features:]
        self.selected_features = [feature_names[i] for i in top_indices]
        
        print(f"Selected {len(self.selected_features)} features based on combined scoring:")
        for feat, score in zip(self.selected_features, combined_scores[top_indices]):
            print(f"  {feat}: {score:.3f}")
        print()
        
        return X[:, top_indices], self.selected_features, combined_scores
    
    def preprocess(self, df, test_size=0.2):
        """Main preprocessing pipeline"""
        print("\n" + "=" * 60)
        print("HEDIS COMPLIANCE PREDICTION PREPROCESSING PIPELINE")
        print("=" * 60 + "\n")
        
        # Feature engineering
        df_eng = self.engineer_features(df)
        
        # Separate features and target
        X = df_eng.drop(columns=[self.target_col]).values
        y = df_eng[self.target_col].values
        self.feature_names = df_eng.drop(columns=[self.target_col]).columns.tolist()
        
        # Check class imbalance
        needs_smote = self.check_class_imbalance(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Test normality and transform
        X_train = self.test_normality(X_train, self.feature_names)
        X_test = self.test_normality(X_test, self.feature_names)
        
        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # Feature selection
        X_train, selected_features, feature_scores = self.select_features(
            X_train, y_train, self.feature_names
        )
        
        # Apply same feature selection to test set
        feature_indices = [self.feature_names.index(f) for f in selected_features]
        X_test = X_test[:, feature_indices]
        
        # Apply SMOTE if needed
        if needs_smote:
            X_train_resampled, y_train_resampled = self.smote.fit_resample(X_train, y_train)
            print(f"SMOTE applied: {len(y_train)} → {len(y_train_resampled)} samples")
            print()
            return X_train_resampled, X_test, y_train_resampled, y_test, feature_scores
        
        return X_train, X_test, y_train, y_test, feature_scores


class HEDISModelEvaluator:
    """Model training and evaluation with professional visualizations"""
    
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.models = {}
        self.results = {}
        
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train Logistic Regression and XGBoost models"""
        print("=" * 60)
        print("MODEL TRAINING")
        print("=" * 60)
        
        # Logistic Regression
        lr = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'
        )
        lr.fit(X_train, y_train)
        self.models['Logistic Regression'] = lr
        
        # XGBoost
        xgb_model = xgb.XGBClassifier(
            random_state=42,
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1])
        )
        xgb_model.fit(X_train, y_train)
        self.models['XGBoost'] = xgb_model
        
        # Evaluate models
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            self.results[name] = {
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            print(f"\n{name} Results:")
            print(f"  ROC-AUC Score: {self.results[name]['roc_auc']:.3f}")
            print(f"  F1-Score: {self.results[name]['classification_report']['weighted avg']['f1-score']:.3f}")
        
        return self.models, self.results
    
    def plot_results(self, X_test, y_test, feature_scores):
        """Create professional visualization dashboard - individual charts"""
        
        # 1. Class Distribution (Original vs SMOTE)
        plt.figure(figsize=(10, 6))
        original_counts = pd.Series(y_test).value_counts()
        bars = plt.bar(original_counts.index, original_counts.values, color=['#FF6B6B', '#4ECDC4'], width=0.6)
        plt.title('Class Distribution (Test Set After SMOTE Training)', fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('HEDIS Compliance Status', fontsize=12)
        plt.ylabel('Number of Members', fontsize=12)
        plt.xticks([0, 1], ['Non-Compliant', 'Compliant'], fontsize=11)
        for bar, value in zip(bars, original_counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{value}\n({value/len(y_test)*100:.1f}%)', 
                    ha='center', fontweight='bold', fontsize=11)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # 2. Feature Importance (Combined Methods)
        plt.figure(figsize=(10, 8))
        top_features = self.preprocessor.selected_features[:10]
        top_scores = feature_scores[[self.preprocessor.feature_names.index(f) 
                                     for f in top_features]]
        y_pos = np.arange(len(top_features))
        bars = plt.barh(y_pos, top_scores, color='#95E77E', height=0.7)
        plt.yticks(y_pos, top_features, fontsize=11)
        plt.xlabel('Importance Score', fontsize=12)
        plt.title('Top 10 Feature Importance (Combined F-test, Mutual Info, Random Forest)', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.grid(axis='x', alpha=0.3)
        for bar, score in zip(bars, top_scores):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{score:.3f}', va='center', fontsize=10)
        plt.tight_layout()
        plt.show()
        
        # 3. ROC Curves Comparison
        plt.figure(figsize=(10, 8))
        colors = ['#FF6B6B', '#4ECDC4']
        for (name, color) in zip(self.models.keys(), colors):
            fpr, tpr, _ = roc_curve(y_test, self.results[name]['y_pred_proba'])
            auc = self.results[name]['roc_auc']
            plt.plot(fpr, tpr, linewidth=3, label=f'{name} (AUC = {auc:.3f})', color=color)
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=2, label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Model Discrimination Comparison', fontsize=14, fontweight='bold', pad=20)
        plt.legend(loc='lower right', fontsize=11, frameon=True, shadow=True)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # 4. Precision-Recall Curves
        plt.figure(figsize=(10, 8))
        colors = ['#FF6B6B', '#4ECDC4']
        for (name, color) in zip(self.models.keys(), colors):
            precision, recall, _ = precision_recall_curve(y_test, self.results[name]['y_pred_proba'])
            plt.plot(recall, precision, linewidth=3, label=name, color=color)
        plt.xlabel('Recall (Sensitivity)', fontsize=12)
        plt.ylabel('Precision (PPV)', fontsize=12)
        plt.title('Precision-Recall Curves - Imbalanced Class Performance', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.legend(loc='lower left', fontsize=11, frameon=True, shadow=True)
        plt.grid(alpha=0.3)
        plt.axhline(y=sum(y_test)/len(y_test), color='gray', linestyle='--', 
                   label='Baseline', alpha=0.5)
        plt.tight_layout()
        plt.show()
        
        # 5. Confusion Matrix - Logistic Regression
        plt.figure(figsize=(8, 6))
        cm_lr = confusion_matrix(y_test, self.results['Logistic Regression']['y_pred'])
        sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', cbar=True,
                   xticklabels=['Non-Compliant', 'Compliant'],
                   yticklabels=['Non-Compliant', 'Compliant'],
                   annot_kws={'size': 14, 'weight': 'bold'})
        plt.title('Confusion Matrix - Logistic Regression', fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        plt.show()
        
        # 6. Confusion Matrix - XGBoost
        plt.figure(figsize=(8, 6))
        cm_xgb = confusion_matrix(y_test, self.results['XGBoost']['y_pred'])
        sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Greens', cbar=True,
                   xticklabels=['Non-Compliant', 'Compliant'],
                   yticklabels=['Non-Compliant', 'Compliant'],
                   annot_kws={'size': 14, 'weight': 'bold'})
        plt.title('Confusion Matrix - XGBoost', fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        plt.show()
        
        # 7. Model Performance Comparison
        plt.figure(figsize=(10, 6))
        metrics = ['Precision', 'Recall', 'F1-Score']
        x = np.arange(len(metrics))
        width = 0.35
        
        lr_scores = [self.results['Logistic Regression']['classification_report']['weighted avg'][m.lower().replace('-', '-')] 
                    for m in ['precision', 'recall', 'f1-score']]
        xgb_scores = [self.results['XGBoost']['classification_report']['weighted avg'][m.lower().replace('-', '-')] 
                     for m in ['precision', 'recall', 'f1-score']]
        
        bars1 = plt.bar(x - width/2, lr_scores, width, label='Logistic Regression', 
                       color='#FF6B6B', alpha=0.8)
        bars2 = plt.bar(x + width/2, xgb_scores, width, label='XGBoost', 
                       color='#4ECDC4', alpha=0.8)
        
        plt.xlabel('Performance Metrics', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=20)
        plt.xticks(x, metrics, fontsize=11)
        plt.legend(fontsize=11, frameon=True, shadow=True)
        plt.grid(axis='y', alpha=0.3)
        plt.ylim([0, 1])
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # 8. XGBoost Feature Importance
        if 'XGBoost' in self.models:
            plt.figure(figsize=(10, 8))
            xgb_importance = self.models['XGBoost'].feature_importances_
            top_xgb_idx = np.argsort(xgb_importance)[-10:]
            top_xgb_features = [self.preprocessor.selected_features[i] for i in top_xgb_idx]
            top_xgb_scores = xgb_importance[top_xgb_idx]
            
            y_pos = np.arange(len(top_xgb_features))
            bars = plt.barh(y_pos, top_xgb_scores, color='#FFE66D', height=0.7)
            plt.yticks(y_pos, top_xgb_features, fontsize=11)
            plt.xlabel('Importance Score', fontsize=12)
            plt.title('XGBoost Feature Importance (Top 10 Tree-Based Splits)', 
                     fontsize=14, fontweight='bold', pad=20)
            plt.grid(axis='x', alpha=0.3)
            for bar, score in zip(bars, top_xgb_scores):
                plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                        f'{score:.3f}', va='center', fontsize=10)
            plt.tight_layout()
            plt.show()
        
        # Print detailed classification reports
        print("\n" + "=" * 60)
        print("DETAILED CLASSIFICATION REPORTS")
        print("=" * 60)
        
        for name in self.models.keys():
            print(f"\n{name}:")
            print(classification_report(y_test, self.results[name]['y_pred'],
                                       target_names=['Non-Compliant', 'Compliant']))


# Main execution
def main():
    # Generate sample data
    np.random.seed(42)
    n = 100
    data = {
        "age": np.random.randint(0, 100, n),
        "racetype": np.random.randint(1, 5, n),
        "ethnicitytype": np.random.randint(1, 25, n),
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
    
    # Initialize preprocessor
    preprocessor = HEDISPreprocessor(target_col='numercnt')
    
    # Preprocess data
    X_train, X_test, y_train, y_test, feature_scores = preprocessor.preprocess(df)
    
    # Initialize evaluator
    evaluator = HEDISModelEvaluator(preprocessor)
    
    # Train models
    models, results = evaluator.train_models(X_train, X_test, y_train, y_test)
    
    # Plot results
    evaluator.plot_results(X_test, y_test, feature_scores)
    
    return preprocessor, models, results

if __name__ == "__main__":
    preprocessor, models, results = main()