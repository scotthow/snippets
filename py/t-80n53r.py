import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, precision_recall_curve, f1_score, make_scorer,
                           balanced_accuracy_score, matthews_corrcoef)
from sklearn.feature_selection import SelectKBest, f_classif, RFE, mutual_info_classif
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier
from scipy import stats
from scipy.stats import normaltest, skew, kurtosis
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set style for professional charts
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ImprovedHEDISPreprocessor:
    """
    Enhanced preprocessing pipeline with multiple improvements for better minority class prediction
    """
    
    def __init__(self, target_col='numercnt', random_state=42):
        self.target_col = target_col
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.power_transformer = PowerTransformer(method='yeo-johnson')
        self.feature_names = None
        self.selected_features = None
        # Try different sampling strategies
        self.sampling_strategies = {
            'SMOTE': SMOTE(random_state=random_state, k_neighbors=3),
            'BorderlineSMOTE': BorderlineSMOTE(random_state=random_state, k_neighbors=3),
            'ADASYN': ADASYN(random_state=random_state, n_neighbors=3),
            'SMOTEENN': SMOTEENN(random_state=random_state),
            'SMOTETomek': SMOTETomek(random_state=random_state)
        }
        self.best_sampler = None
        self.normality_results = {}
        self.log_transformed_cols = []
        self.power_transformed_cols = []
        self.original_train_distribution = None
        self.balanced_train_distribution = None
        
    def engineer_advanced_features(self, df):
        """Create more sophisticated engineered features"""
        df_eng = df.copy()
        
        # Original composite scores
        df_eng['social_vulnerability'] = (
            df_eng['poverty_score'] * 0.3 +
            df_eng['social_isolation'] * 0.2 +
            df_eng['unemployment_index'] * 0.2 +
            (100 - df_eng['education_index']) * 0.15 +
            (100 - df_eng['income_index']) * 0.15
        )
        
        df_eng['access_composite'] = (
            df_eng['food_access'] * 0.25 +
            df_eng['health_access'] * 0.35 +
            df_eng['transport_access'] * 0.25 +
            df_eng['technology_access'] * 0.15
        )
        
        df_eng['housing_stability'] = (
            df_eng['housing_ownership'] * 0.5 +
            (100 - df_eng['housing_desert']) * 0.5
        )
        
        df_eng['environmental_risk'] = (
            df_eng['natural_disaster'] * 0.6 +
            (1 - df_eng['water_quality']) * 100 * 0.4
        )
        
        # NEW: Advanced features for minority class detection
        
        # Risk stratification score
        df_eng['risk_score'] = (
            (df_eng['age'] / 100) * 20 +
            (df_eng['poverty_score'] / 100) * 25 +
            (df_eng['social_isolation'] / 100) * 20 +
            ((100 - df_eng['health_access']) / 100) * 35
        )
        
        # Barrier index - identifies members with multiple barriers
        df_eng['barrier_count'] = (
            (df_eng['poverty_score'] > 70).astype(int) +
            (df_eng['transport_access'] < 30).astype(int) +
            (df_eng['health_access'] < 30).astype(int) +
            (df_eng['technology_access'] < 30).astype(int) +
            (df_eng['social_isolation'] > 70).astype(int)
        )
        
        # Engagement potential score
        df_eng['engagement_potential'] = (
            df_eng['technology_access'] * 0.3 +
            df_eng['education_index'] * 0.3 +
            (100 - df_eng['social_isolation']) * 0.2 +
            df_eng['transport_access'] * 0.2
        )
        
        # Age-specific risk categories with proper handling
        # Use numpy.digitize for more robust binning
        age_bins = [0, 25, 40, 55, 65, 75, 100]
        df_eng['age_risk'] = np.digitize(df_eng['age'], bins=age_bins)
        
        # Polynomial features for key predictors
        df_eng['age_squared'] = df_eng['age'] ** 2
        df_eng['poverty_squared'] = df_eng['poverty_score'] ** 2
        
        # Interaction terms for high-impact combinations
        df_eng['age_health_interaction'] = df_eng['age'] * df_eng['health_access'] / 100
        df_eng['poverty_access_interaction'] = df_eng['poverty_score'] * df_eng['access_composite'] / 100
        df_eng['social_barrier_interaction'] = df_eng['social_vulnerability'] * df_eng['barrier_count']
        
        # Ratio features (add small constant to avoid division by zero)
        df_eng['access_to_risk_ratio'] = (df_eng['access_composite'] + 1) / (df_eng['risk_score'] + 1)
        df_eng['engagement_to_barrier_ratio'] = (df_eng['engagement_potential'] + 1) / (df_eng['barrier_count'] + 1)
        
        print(f"✓ Advanced feature engineering completed - Added {len(df_eng.columns) - len(df.columns)} features")
        
        return df_eng
    
    def select_best_sampler(self, X_train, y_train):
        """Test different sampling strategies and select the best one"""
        print("\nTesting sampling strategies...")
        best_score = -1
        best_sampler_name = None
        
        for name, sampler in self.sampling_strategies.items():
            try:
                X_temp, y_temp = sampler.fit_resample(X_train[:min(50, len(X_train))], 
                                                       y_train[:min(50, len(y_train))])
                # Quick test with logistic regression
                lr = LogisticRegression(random_state=self.random_state, max_iter=100)
                scores = cross_val_score(lr, X_temp, y_temp, cv=3, 
                                        scoring='balanced_accuracy', n_jobs=-1)
                mean_score = np.mean(scores)
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_sampler_name = name
                    
            except Exception as e:
                print(f"  {name} failed: {str(e)}")
                continue
        
        self.best_sampler = self.sampling_strategies[best_sampler_name]
        print(f"  Selected: {best_sampler_name} (score: {best_score:.3f})")
        return self.best_sampler
    
    def preprocess(self, df, test_size=0.2):
        """Enhanced preprocessing pipeline"""
        print("\n" + "=" * 60)
        print("ENHANCED HEDIS COMPLIANCE PREDICTION PIPELINE")
        print("=" * 60 + "\n")
        
        # Advanced feature engineering
        df_eng = self.engineer_advanced_features(df)
        
        # Separate features and target
        X = df_eng.drop(columns=[self.target_col]).values
        y = df_eng[self.target_col].values
        self.feature_names = df_eng.drop(columns=[self.target_col]).columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Store original distribution
        unique, counts = np.unique(y_train, return_counts=True)
        self.original_train_distribution = dict(zip(unique, counts))
        
        # Scale features BEFORE selection for better performance
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # Enhanced feature selection with more features
        X_train, selected_features, feature_scores = self.select_features_enhanced(
            X_train, y_train, self.feature_names, n_features=20  # Increased from 15
        )
        
        # Apply same feature selection to test set
        feature_indices = [self.feature_names.index(f) for f in selected_features]
        X_test = X_test[:, feature_indices]
        
        # Select and apply best sampling strategy
        best_sampler = self.select_best_sampler(X_train, y_train)
        X_train_resampled, y_train_resampled = best_sampler.fit_resample(X_train, y_train)
        
        # Store balanced distribution
        unique, counts = np.unique(y_train_resampled, return_counts=True)
        self.balanced_train_distribution = dict(zip(unique, counts))
        
        print(f"\nSampling applied: {len(y_train)} → {len(y_train_resampled)} samples")
        print(f"  Original: Class 0: {self.original_train_distribution[0]}, Class 1: {self.original_train_distribution[1]}")
        print(f"  Balanced: Class 0: {self.balanced_train_distribution[0]}, Class 1: {self.balanced_train_distribution[1]}")
        
        return X_train_resampled, X_test, y_train_resampled, y_test, feature_scores
    
    def select_features_enhanced(self, X, y, feature_names, n_features=20):
        """Enhanced feature selection with focus on minority class"""
        print("\n" + "=" * 60)
        print("ENHANCED FEATURE SELECTION")
        print("=" * 60)
        
        # Method 1: F-test
        selector_f = SelectKBest(f_classif, k=min(n_features, X.shape[1]))
        selector_f.fit(X, y)
        f_scores = selector_f.scores_
        
        # Method 2: Mutual Information
        mi_scores = mutual_info_classif(X, y, random_state=self.random_state)
        
        # Method 3: Balanced Random Forest for better minority class importance
        brf = BalancedRandomForestClassifier(n_estimators=100, random_state=self.random_state)
        brf.fit(X, y)
        rf_importance = brf.feature_importances_
        
        # Normalize and combine with adjusted weights
        f_scores_norm = f_scores / (f_scores.max() + 1e-10)
        mi_scores_norm = mi_scores / (mi_scores.max() + 1e-10)
        rf_scores_norm = rf_importance / (rf_importance.max() + 1e-10)
        
        # Give more weight to balanced random forest
        combined_scores = (f_scores_norm * 0.25 + mi_scores_norm * 0.25 + rf_scores_norm * 0.5)
        
        # Select top features
        top_indices = np.argsort(combined_scores)[-n_features:]
        self.selected_features = [feature_names[i] for i in top_indices]
        
        print(f"Selected {len(self.selected_features)} features with minority-class focus")
        
        return X[:, top_indices], self.selected_features, combined_scores


class EnhancedModelTrainer:
    """Advanced model training with multiple improvements"""
    
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.models = {}
        self.results = {}
        
    def create_models(self):
        """Create ensemble of models optimized for imbalanced data"""
        
        # 1. Balanced Random Forest
        brf = BalancedRandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
        
        # 2. XGBoost with better parameters for imbalanced data
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=5,  # Increased weight for minority class
            min_child_weight=5,
            gamma=1,
            random_state=42
        )
        
        # 3. Calibrated Logistic Regression
        lr = LogisticRegression(
            class_weight='balanced',
            C=0.1,  # More regularization
            max_iter=1000,
            random_state=42
        )
        calibrated_lr = CalibratedClassifierCV(lr, cv=3, method='sigmoid')
        
        # 4. Gradient Boosting with balanced subsample
        gb = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=42
        )
        
        # 5. Voting Classifier (Ensemble)
        voting = VotingClassifier(
            estimators=[
                ('brf', brf),
                ('xgb', xgb_model),
                ('lr', calibrated_lr)
            ],
            voting='soft',
            weights=[2, 1, 1]  # Give more weight to balanced random forest
        )
        
        return {
            'Balanced Random Forest': brf,
            'XGBoost (Tuned)': xgb_model,
            'Calibrated Logistic Regression': calibrated_lr,
            'Gradient Boosting': gb,
            'Voting Ensemble': voting
        }
    
    def optimize_threshold(self, model, X_val, y_val):
        """Find optimal classification threshold for better minority class recall"""
        y_proba = model.predict_proba(X_val)[:, 1]
        
        best_threshold = 0.5
        best_f1 = 0
        
        for threshold in np.arange(0.1, 0.9, 0.05):
            y_pred = (y_proba >= threshold).astype(int)
            f1 = f1_score(y_val, y_pred)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        return best_threshold
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Train models with cross-validation and threshold optimization"""
        print("\n" + "=" * 60)
        print("ENHANCED MODEL TRAINING & EVALUATION")
        print("=" * 60)
        
        # Create validation set for threshold tuning
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        models = self.create_models()
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train_split, y_train_split)
            
            # Find optimal threshold
            threshold = self.optimize_threshold(model, X_val, y_val)
            
            # Retrain on full training set
            model.fit(X_train, y_train)
            
            # Predictions with optimized threshold
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            # Calculate metrics
            self.results[name] = {
                'model': model,
                'threshold': threshold,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
                'matthews_corrcoef': matthews_corrcoef(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            # Print key metrics
            cr = self.results[name]['classification_report']
            print(f"  Threshold: {threshold:.2f}")
            print(f"  Minority Class Recall: {cr['1']['recall']:.3f}")
            print(f"  Minority Class Precision: {cr['1']['precision']:.3f}")
            print(f"  Balanced Accuracy: {self.results[name]['balanced_accuracy']:.3f}")
            print(f"  Matthews Correlation: {self.results[name]['matthews_corrcoef']:.3f}")
        
        # Select best model based on minority class F1-score
        best_model = max(self.results.items(), 
                        key=lambda x: x[1]['classification_report']['1']['f1-score'])
        print(f"\n🏆 Best Model: {best_model[0]} (Minority F1: {best_model[1]['classification_report']['1']['f1-score']:.3f})")
        
        return self.results
    
    def plot_enhanced_results(self, y_test):
        """Create enhanced visualizations focusing on minority class performance"""
        
        # 1. Class Distribution Pie Charts (Before and After SMOTE)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Original distribution pie chart
        if self.preprocessor.original_train_distribution:
            labels = ['Non-Compliant', 'Compliant']
            sizes_orig = [self.preprocessor.original_train_distribution[0], 
                         self.preprocessor.original_train_distribution[1]]
            colors = ['#FF6B6B', '#4ECDC4']
            explode = (0.05, 0.05)
            
            # Original distribution
            wedges1, texts1, autotexts1 = ax1.pie(sizes_orig, explode=explode, labels=labels, 
                                                   colors=colors, autopct='%1.1f%%',
                                                   shadow=True, startangle=90, 
                                                   textprops={'fontsize': 12, 'fontweight': 'bold'})
            ax1.set_title('Original Training Set\n(Before SMOTE)', 
                         fontsize=14, fontweight='bold', pad=20)
            
            # Add count labels
            total_orig = sum(sizes_orig)
            ax1.text(0, -1.3, f'Total: {total_orig} samples\nClass 0: {sizes_orig[0]} | Class 1: {sizes_orig[1]}', 
                    ha='center', fontsize=11)
        
        # Balanced distribution pie chart (After SMOTE)
        if self.preprocessor.balanced_train_distribution:
            sizes_balanced = [self.preprocessor.balanced_train_distribution[0], 
                            self.preprocessor.balanced_train_distribution[1]]
            
            wedges2, texts2, autotexts2 = ax2.pie(sizes_balanced, explode=explode, labels=labels, 
                                                   colors=colors, autopct='%1.1f%%',
                                                   shadow=True, startangle=90, 
                                                   textprops={'fontsize': 12, 'fontweight': 'bold'})
            ax2.set_title('Balanced Training Set\n(After SMOTE)', 
                         fontsize=14, fontweight='bold', pad=20)
            
            # Add count labels
            total_balanced = sum(sizes_balanced)
            ax2.text(0, -1.3, f'Total: {total_balanced} samples\nClass 0: {sizes_balanced[0]} | Class 1: {sizes_balanced[1]}', 
                    ha='center', fontsize=11)
        
        plt.suptitle('Training Set Class Distribution: Impact of SMOTE Balancing', 
                    fontsize=16, fontweight='bold', y=1.05)
        plt.tight_layout()
        plt.show()
        
        # 2. ROC-AUC Curves for All Models
        plt.figure(figsize=(12, 8))
        
        # Color palette for different models
        colors_palette = ['#FF6B6B', '#4ECDC4', '#95E77E', '#FFE66D', '#B4A7D6']
        
        for idx, (name, color) in enumerate(zip(self.results.keys(), colors_palette)):
            fpr, tpr, thresholds = roc_curve(y_test, self.results[name]['y_pred_proba'])
            auc = self.results[name]['roc_auc']
            
            # Plot ROC curve with thicker lines
            plt.plot(fpr, tpr, linewidth=2.5, label=f'{name} (AUC = {auc:.3f})', 
                    color=color, alpha=0.8)
            
            # Mark the optimal threshold point
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', s=100, 
                       color=color, edgecolors='black', linewidth=2, zorder=5)
            
            # Annotate optimal points for top 3 models
            if idx < 3:
                plt.annotate(f'Optimal\n({fpr[optimal_idx]:.2f}, {tpr[optimal_idx]:.2f})',
                           xy=(fpr[optimal_idx], tpr[optimal_idx]),
                           xytext=(fpr[optimal_idx] + 0.1, tpr[optimal_idx] - 0.1),
                           fontsize=9, ha='center',
                           arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
        
        # Plot diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=2, label='Random Classifier')
        
        # Formatting
        plt.xlabel('False Positive Rate', fontsize=13)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=13)
        plt.title('ROC Curves - Model Comparison', fontsize=15, fontweight='bold', pad=20)
        plt.legend(loc='lower right', fontsize=11, frameon=True, shadow=True)
        plt.grid(alpha=0.3, linestyle='--')
        
        # Add shaded area for excellent performance zone
        plt.fill_between([0, 0, 0.2], [0.8, 1, 1], [1, 1, 1], 
                        alpha=0.1, color='green', label='Excellent Zone')
        
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.tight_layout()
        plt.show()
        
        # 3. Minority Class Performance Comparison
        plt.figure(figsize=(12, 6))
        models = list(self.results.keys())
        minority_metrics = ['precision', 'recall', 'f1-score']
        
        x = np.arange(len(models))
        width = 0.25
        
        for i, metric in enumerate(minority_metrics):
            values = [self.results[m]['classification_report']['1'][metric] for m in models]
            plt.bar(x + i*width, values, width, label=metric.capitalize(), alpha=0.8)
        
        plt.xlabel('Models', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Minority Class (Compliant) Performance Comparison', fontsize=14, fontweight='bold')
        plt.xticks(x + width, models, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # 2. Balanced Metrics Comparison
        plt.figure(figsize=(10, 6))
        balanced_metrics = {
            'Balanced\nAccuracy': [self.results[m]['balanced_accuracy'] for m in models],
            'Matthews\nCorrelation': [self.results[m]['matthews_corrcoef'] for m in models],
            'ROC-AUC': [self.results[m]['roc_auc'] for m in models]
        }
        
        x = np.arange(len(models))
        width = 0.25
        
        for i, (metric, values) in enumerate(balanced_metrics.items()):
            plt.bar(x + i*width, values, width, label=metric, alpha=0.8)
        
        plt.xlabel('Models', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Balanced Performance Metrics', fontsize=14, fontweight='bold')
        plt.xticks(x + width, models, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # 3. Optimal Thresholds
        plt.figure(figsize=(10, 6))
        thresholds = [self.results[m]['threshold'] for m in models]
        bars = plt.bar(models, thresholds, color='#4ECDC4', alpha=0.7)
        plt.axhline(y=0.5, color='red', linestyle='--', label='Default (0.5)')
        plt.xlabel('Models', fontsize=12)
        plt.ylabel('Optimal Threshold', fontsize=12)
        plt.title('Optimized Classification Thresholds', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, thresholds):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # 4. Best Model Confusion Matrix
        best_model_name = max(self.results.items(), 
                             key=lambda x: x[1]['classification_report']['1']['f1-score'])[0]
        
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, self.results[best_model_name]['y_pred'])
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create annotation text with both count and percentage
        annot_text = np.array([[f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)' 
                                for j in range(2)] for i in range(2)])
        
        sns.heatmap(cm, annot=annot_text, fmt='', cmap='YlOrRd',
                   xticklabels=['Non-Compliant', 'Compliant'],
                   yticklabels=['Non-Compliant', 'Compliant'],
                   cbar_kws={'label': 'Count'})
        
        plt.title(f'Best Model Confusion Matrix: {best_model_name}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        plt.tight_layout()
        plt.show()


def main_improved():
    """Main execution with improvements"""
    # Generate sample data
    np.random.seed(42)
    n = 500  # Increased sample size for better training
    
    # Create more realistic imbalanced data with patterns
    data = {
        "age": np.random.randint(0, 100, n),
        "racetype": np.random.randint(1, 5, n),
        "ethnicitytype": np.random.randint(1, 25, n),
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
    
    # Create target with some correlation to features
    df = pd.DataFrame(data)
    
    # Make compliance more likely with better access and lower poverty
    compliance_score = (
        df['health_access'] * 0.3 + 
        df['education_index'] * 0.2 + 
        (100 - df['poverty_score']) * 0.3 +
        df['transport_access'] * 0.2 +
        np.random.uniform(-20, 20, n)  # Add noise
    )
    
    # Create imbalanced target (15% positive class)
    threshold = np.percentile(compliance_score, 85)
    df['numercnt'] = (compliance_score > threshold).astype(int)
    
    print("Target distribution:")
    print(df['numercnt'].value_counts(normalize=True))
    
    # Initialize improved preprocessor
    preprocessor = ImprovedHEDISPreprocessor(target_col='numercnt')
    
    # Preprocess data
    X_train, X_test, y_train, y_test, feature_scores = preprocessor.preprocess(df)
    
    # Initialize enhanced trainer
    trainer = EnhancedModelTrainer(preprocessor)
    
    # Train and evaluate models
    results = trainer.train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # Plot enhanced results
    trainer.plot_enhanced_results(y_test)
    
    # Print detailed report for best model
    best_model_name = max(results.items(), 
                         key=lambda x: x[1]['classification_report']['1']['f1-score'])[0]
    
    print("\n" + "=" * 60)
    print(f"BEST MODEL DETAILED REPORT: {best_model_name}")
    print("=" * 60)
    print("\nClassification Report:")
    print(classification_report(y_test, results[best_model_name]['y_pred'],
                               target_names=['Non-Compliant', 'Compliant']))
    
    return preprocessor, trainer, results

if __name__ == "__main__":
    preprocessor, trainer, results = main_improved()