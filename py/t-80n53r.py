import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif, RFE, RFECV
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class HEDISPreprocessor:
    """Preprocessing pipeline for HEDIS compliance prediction"""
    
    def __init__(self, target_col='numercnt', test_size=0.2, random_state=42):
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.feature_encoders = {}
        self.scaler = StandardScaler()
        self.selected_features = None
        self.feature_importance = None
        
    def initial_clean(self, df):
        """Initial data cleaning and type conversion"""
        df = df.copy()
        
        # Convert date columns to datetime
        date_cols = ['clssdt', 'bus_eff_dt', 'max_source_pstd_dts']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Ensure numeric columns are properly typed
        numeric_cols = ['zip_code', 'bgfips', 'block_code']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def create_temporal_features(self, df):
        """Create time-based features from date columns"""
        df = df.copy()
        
        # Reference date (use max date in dataset or current date)
        if 'max_source_pstd_dts' in df.columns:
            ref_date = df['max_source_pstd_dts'].max()
            if pd.isna(ref_date):
                ref_date = pd.Timestamp.now()
        else:
            ref_date = pd.Timestamp.now()
        
        # Days since class date (HEDIS measurement)
        if 'clssdt' in df.columns:
            df['days_since_measurement'] = (ref_date - df['clssdt']).dt.days
            df['measurement_month'] = df['clssdt'].dt.month
            df['measurement_quarter'] = df['clssdt'].dt.quarter
        
        # Days since business effective date
        if 'bus_eff_dt' in df.columns:
            df['days_since_bus_eff'] = (ref_date - df['bus_eff_dt']).dt.days
            df['membership_tenure_days'] = df['days_since_bus_eff']
        
        # Age groups for better modeling
        if 'age' in df.columns:
            df['age_group'] = pd.cut(df['age'], 
                                     bins=[0, 18, 35, 50, 65, 80, 120],
                                     labels=['<18', '18-34', '35-49', '50-64', '65-79', '80+'])
        
        return df
    
    def create_sdoh_indices(self, df):
        """Create composite Social Determinants of Health indices"""
        df = df.copy()
        
        # Economic hardship index
        econ_cols = ['poverty_score', 'unemployment_index', 'income_inequality']
        econ_cols = [c for c in econ_cols if c in df.columns]
        if econ_cols:
            df['economic_hardship_index'] = df[econ_cols].mean(axis=1)
        
        # Access barrier index
        access_cols = ['food_access', 'health_access', 'transport_access', 'technology_access']
        access_cols = [c for c in access_cols if c in df.columns]
        if access_cols:
            df['access_barrier_index'] = df[access_cols].mean(axis=1)
        
        # Housing vulnerability index
        housing_cols = ['housing_desert', 'housing_ownership']
        housing_cols = [c for c in housing_cols if c in df.columns]
        if housing_cols:
            df['housing_vulnerability_index'] = df[housing_cols].mean(axis=1)
        
        # Environmental risk index
        env_cols = ['natural_disaster', 'water_quality']
        env_cols = [c for c in env_cols if c in df.columns]
        if env_cols:
            df['environmental_risk_index'] = df[env_cols].mean(axis=1)
        
        # Overall vulnerability score (weighted)
        if 'social_risk_score' in df.columns:
            df['overall_vulnerability'] = df['social_risk_score']
        else:
            vulnerability_indices = [
                'economic_hardship_index', 'access_barrier_index', 
                'housing_vulnerability_index', 'environmental_risk_index'
            ]
            existing_indices = [c for c in vulnerability_indices if c in df.columns]
            if existing_indices:
                df['overall_vulnerability'] = df[existing_indices].mean(axis=1)
        
        return df
    
    def create_geographic_features(self, df):
        """Create geographic and spatial features"""
        df = df.copy()
        
        # State-level aggregations (if applicable)
        if 'state' in df.columns and self.target_col in df.columns:
            state_compliance = df.groupby('state')[self.target_col].mean()
            df['state_compliance_rate'] = df['state'].map(state_compliance)
        
        # ZIP code level features
        if 'zip_code' in df.columns:
            df['zip_prefix'] = df['zip_code'].astype(str).str[:3]
            if self.target_col in df.columns:
                zip_compliance = df.groupby('zip_prefix')[self.target_col].mean()
                df['zip_area_compliance_rate'] = df['zip_prefix'].map(zip_compliance)
        
        # Distance from major cities (if lat/lon available)
        if 'latitude' in df.columns and 'longitude' in df.columns:
            # Calculate distance to nearest healthcare facility cluster
            # This is simplified - in production, you'd use actual facility locations
            df['lat_lon_cluster'] = (
                df['latitude'].round(1).astype(str) + '_' + 
                df['longitude'].round(1).astype(str)
            )
        
        return df
    
    def create_measure_features(self, df):
        """Create features related to HEDIS measures"""
        df = df.copy()
        
        # Measure type indicators
        if 'measure_tla' in df.columns:
            # Common HEDIS measure categories
            df['is_prevention_measure'] = df['measure_tla'].str.contains(
                'BCS|CCS|COL|IMM|W15|W34|WCV', case=False, na=False
            ).astype(int)
            df['is_chronic_care'] = df['measure_tla'].str.contains(
                'CDC|CBP|SPD|CMC|SPC', case=False, na=False
            ).astype(int)
            df['is_behavioral_health'] = df['measure_tla'].str.contains(
                'AMM|FUH|FUM|SAA|SMD', case=False, na=False
            ).astype(int)
        
        # Customer segment features
        if 'cust_seg_cd' in df.columns:
            df['is_commercial'] = df['cust_seg_cd'].str.contains(
                'COM|COMM', case=False, na=False
            ).astype(int)
            df['is_medicare'] = df['cust_seg_cd'].str.contains(
                'MED|MCARE', case=False, na=False
            ).astype(int)
            df['is_medicaid'] = df['cust_seg_cd'].str.contains(
                'MCAID|MCD', case=False, na=False
            ).astype(int)
        
        return df
    
    def encode_categorical(self, df, categorical_cols=None):
        """Encode categorical variables"""
        df = df.copy()
        
        if categorical_cols is None:
            categorical_cols = [
                'racetype', 'ethnicitytype', 'measure_tla', 'submeasurekey',
                'cust_seg_cd', 'lob_cd', 'state', 'age_group'
            ]
        
        categorical_cols = [c for c in categorical_cols if c in df.columns]
        
        for col in categorical_cols:
            if col not in self.feature_encoders:
                # Use LabelEncoder for binary/ordinal or OneHot for nominal
                n_unique = df[col].nunique()
                
                if n_unique <= 2:
                    # Binary encoding
                    self.feature_encoders[col] = LabelEncoder()
                    df[col + '_encoded'] = self.feature_encoders[col].fit_transform(
                        df[col].fillna('missing')
                    )
                elif n_unique <= 10:
                    # One-hot encoding for low cardinality
                    dummies = pd.get_dummies(
                        df[col].fillna('missing'), 
                        prefix=col, 
                        drop_first=True
                    )
                    df = pd.concat([df, dummies], axis=1)
                else:
                    # Target encoding for high cardinality
                    if self.target_col in df.columns:
                        target_mean = df.groupby(col)[self.target_col].mean()
                        df[col + '_target_encoded'] = df[col].map(target_mean)
                        df[col + '_target_encoded'].fillna(
                            df[self.target_col].mean(), inplace=True
                        )
            else:
                # Use existing encoder
                if isinstance(self.feature_encoders[col], LabelEncoder):
                    df[col + '_encoded'] = self.feature_encoders[col].transform(
                        df[col].fillna('missing')
                    )
        
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values strategically"""
        df = df.copy()
        
        # Numeric columns - use median or indicator
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != self.target_col:
                missing_pct = df[col].isna().mean()
                
                if missing_pct > 0.3:
                    # Create missing indicator for high missingness
                    df[col + '_is_missing'] = df[col].isna().astype(int)
                
                if missing_pct < 0.5:
                    # Impute with median for moderate missingness
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    # Fill with 0 for very high missingness (likely not measured)
                    df[col].fillna(0, inplace=True)
        
        # Categorical columns - fill with 'missing'
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            df[col].fillna('missing', inplace=True)
        
        return df
    
    def select_features(self, X, y, method='all', n_features=30):
        """Feature selection using multiple methods"""
        
        # Remove non-numeric columns for modeling
        X_numeric = X.select_dtypes(include=[np.number])
        
        results = {}
        
        # 1. Univariate Selection (F-statistic)
        if method in ['all', 'univariate']:
            selector_f = SelectKBest(score_func=f_classif, k=min(n_features, X_numeric.shape[1]))
            selector_f.fit(X_numeric, y)
            results['f_scores'] = pd.Series(
                selector_f.scores_, 
                index=X_numeric.columns
            ).sort_values(ascending=False)
        
        # 2. Mutual Information
        if method in ['all', 'mutual_info']:
            selector_mi = SelectKBest(
                score_func=mutual_info_classif, 
                k=min(n_features, X_numeric.shape[1])
            )
            selector_mi.fit(X_numeric, y)
            results['mi_scores'] = pd.Series(
                selector_mi.scores_, 
                index=X_numeric.columns
            ).sort_values(ascending=False)
        
        # 3. Recursive Feature Elimination with Logistic Regression
        if method in ['all', 'rfe']:
            lr = LogisticRegression(random_state=self.random_state, max_iter=1000)
            rfe = RFE(lr, n_features_to_select=min(n_features, X_numeric.shape[1]))
            rfe.fit(X_numeric, y)
            results['rfe_ranking'] = pd.Series(
                rfe.ranking_, 
                index=X_numeric.columns
            ).sort_values()
        
        # 4. Random Forest Feature Importance
        if method in ['all', 'rf']:
            rf = RandomForestClassifier(
                n_estimators=100, 
                random_state=self.random_state,
                n_jobs=-1
            )
            rf.fit(X_numeric, y)
            results['rf_importance'] = pd.Series(
                rf.feature_importances_, 
                index=X_numeric.columns
            ).sort_values(ascending=False)
        
        # Combine results to get consensus top features
        if method == 'all':
            feature_scores = pd.DataFrame()
            
            for key in results:
                if 'ranking' in key:
                    # Lower ranking is better
                    feature_scores[key] = results[key].rank()
                else:
                    # Higher score is better
                    feature_scores[key] = results[key].rank(ascending=False)
            
            feature_scores['avg_rank'] = feature_scores.mean(axis=1)
            self.selected_features = feature_scores.nsmallest(
                n_features, 'avg_rank'
            ).index.tolist()
        else:
            # Use single method result
            self.selected_features = results[list(results.keys())[0]].head(n_features).index.tolist()
        
        self.feature_importance = results
        return self.selected_features
    
    def preprocess_pipeline(self, df, fit=True):
        """Complete preprocessing pipeline"""
        
        # 1. Initial cleaning
        df = self.initial_clean(df)
        
        # 2. Create temporal features
        df = self.create_temporal_features(df)
        
        # 3. Create SDOH indices
        df = self.create_sdoh_indices(df)
        
        # 4. Create geographic features
        df = self.create_geographic_features(df)
        
        # 5. Create measure-specific features
        df = self.create_measure_features(df)
        
        # 6. Handle missing values
        df = self.handle_missing_values(df)
        
        # 7. Encode categorical variables
        df = self.encode_categorical(df)
        
        # 8. Drop unnecessary columns
        cols_to_drop = [
            'member_key', 'src_mbr_id', 'asdb_plankey', 'indiv_anlytcs_id',
            'iodb_mbr_key', 'iodb_plan_key', 'cust_no', 'cust_nm',
            'address_line_1', 'city', 'clssdt', 'bus_eff_dt', 
            'max_source_pstd_dts', 'denomcnt'
        ]
        
        # Also drop original categorical columns that were encoded
        encoded_cols = [col for col in df.columns if '_encoded' in col or '_target_encoded' in col]
        if encoded_cols:
            original_cats = [col.replace('_encoded', '').replace('_target_', '') for col in encoded_cols]
            cols_to_drop.extend(original_cats)
        
        cols_to_drop = [c for c in cols_to_drop if c in df.columns]
        df = df.drop(columns=cols_to_drop)
        
        # 9. Separate features and target
        if self.target_col in df.columns:
            X = df.drop(columns=[self.target_col])
            y = df[self.target_col]
            
            # 10. Feature selection (only on training data)
            if fit:
                self.select_features(X, y, method='all', n_features=30)
            
            # Keep only selected features
            if self.selected_features:
                X = X[self.selected_features]
            
            # 11. Scale features
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
        else:
            # For prediction without target
            if self.selected_features:
                df = df[self.selected_features]
            
            df_scaled = pd.DataFrame(
                self.scaler.transform(df),
                columns=df.columns,
                index=df.index
            )
            
            return df_scaled


class HEDISModelPipeline:
    """Complete modeling pipeline for HEDIS compliance prediction"""
    
    def __init__(self, preprocessor=None):
        self.preprocessor = preprocessor or HEDISPreprocessor()
        self.models = {}
        self.best_model = None
        self.results = {}
        
    def build_models(self, X_train, y_train, X_test, y_test, use_smote=True):
        """Build and evaluate multiple models"""
        
        # Handle class imbalance with SMOTE
        if use_smote:
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            print(f"Class distribution after SMOTE:")
            print(pd.Series(y_train_balanced).value_counts())
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
        
        # 1. Logistic Regression
        lr_model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced',
            solver='liblinear'
        )
        lr_model.fit(X_train_balanced, y_train_balanced)
        self.models['logistic_regression'] = lr_model
        
        # Evaluate
        lr_pred = lr_model.predict(X_test)
        lr_pred_proba = lr_model.predict_proba(X_test)[:, 1]
        self.results['logistic_regression'] = {
            'roc_auc': roc_auc_score(y_test, lr_pred_proba),
            'classification_report': classification_report(y_test, lr_pred),
            'confusion_matrix': confusion_matrix(y_test, lr_pred)
        }
        
        # 2. XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective='binary:logistic',
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train_balanced, y_train_balanced)
        self.models['xgboost'] = xgb_model
        
        # Evaluate
        xgb_pred = xgb_model.predict(X_test)
        xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
        self.results['xgboost'] = {
            'roc_auc': roc_auc_score(y_test, xgb_pred_proba),
            'classification_report': classification_report(y_test, xgb_pred),
            'confusion_matrix': confusion_matrix(y_test, xgb_pred)
        }
        
        # 3. XGBoost with hyperparameter tuning
        scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
        xgb_tuned = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            objective='binary:logistic',
            random_state=42,
            use_label_encoder=False,
            eval_metric='auc'
        )
        xgb_tuned.fit(X_train_balanced, y_train_balanced)
        self.models['xgboost_tuned'] = xgb_tuned
        
        # Evaluate
        xgb_tuned_pred = xgb_tuned.predict(X_test)
        xgb_tuned_pred_proba = xgb_tuned.predict_proba(X_test)[:, 1]
        self.results['xgboost_tuned'] = {
            'roc_auc': roc_auc_score(y_test, xgb_tuned_pred_proba),
            'classification_report': classification_report(y_test, xgb_tuned_pred),
            'confusion_matrix': confusion_matrix(y_test, xgb_tuned_pred)
        }
        
        # Select best model based on ROC-AUC
        best_score = 0
        for model_name, result in self.results.items():
            if result['roc_auc'] > best_score:
                best_score = result['roc_auc']
                self.best_model = model_name
        
        return self.results
    
    def display_results(self):
        """Display model evaluation results"""
        print("="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        
        for model_name, result in self.results.items():
            print(f"\n{model_name.upper()}")
            print("-"*40)
            print(f"ROC-AUC Score: {result['roc_auc']:.4f}")
            print("\nClassification Report:")
            print(result['classification_report'])
            print("\nConfusion Matrix:")
            print(result['confusion_matrix'])
        
        print("\n" + "="*60)
        print(f"BEST MODEL: {self.best_model} (ROC-AUC: {self.results[self.best_model]['roc_auc']:.4f})")
        print("="*60)
    
    def get_feature_importance(self, model_name='xgboost_tuned'):
        """Get feature importance from tree-based models"""
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return None
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.preprocessor.selected_features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            print(f"Model {model_name} doesn't have feature_importances_")
            return None


# Example usage
def main():
    """Main execution function"""
    
    # Assuming you have loaded your dataframe as 'df'
    # df = pd.read_csv('your_data.csv')
    
    # Initialize preprocessor
    preprocessor = HEDISPreprocessor(target_col='numercnt')
    
    # Preprocess data
    # X, y = preprocessor.preprocess_pipeline(df, fit=True)
    
    # Split data
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=42, stratify=y
    # )
    
    # Initialize model pipeline
    # model_pipeline = HEDISModelPipeline(preprocessor)
    
    # Build and evaluate models
    # results = model_pipeline.build_models(X_train, y_train, X_test, y_test, use_smote=True)
    
    # Display results
    # model_pipeline.display_results()
    
    # Get feature importance
    # feature_importance = model_pipeline.get_feature_importance('xgboost_tuned')
    # print("\nTop 15 Most Important Features:")
    # print(feature_importance.head(15))
    
    print("Pipeline created successfully! Uncomment the code in main() to run with your data.")


if __name__ == "__main__":
    main()