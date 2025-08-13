import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# For demonstration, let's create a sample DataFrame matching your schema
# In your actual use case, you would load your data, e.g., df = pd.read_csv('your_data.csv')
data = {
    'measurement_year': [2023] * 100, 'member_key': range(100), 'denomcnt': [1] * 100,
    'numercnt': np.random.choice([0, 1], 100, p=[0.8, 0.2]), # Imbalanced target
    'age': np.random.randint(40, 75, 100),
    'racetype': np.random.randint(1, 6, 100), 'ethnicitytype': np.random.randint(1, 3, 100),
    'measure_tla': np.random.choice(['BCS', 'CWP', 'AAB'], 100),
    'state': np.random.choice(['CA', 'NY', 'TX'], 100),
    'food_access': np.random.rand(100), 'health_access': np.random.rand(100),
    'poverty_score': np.random.rand(100), 'unemployment_index': np.random.rand(100),
    'income_index': np.random.rand(100), 'social_risk_score': np.random.rand(100)
}
# Add other columns from the schema with dummy data if needed
df = pd.DataFrame(data)
# Create a highly correlated feature to demonstrate removal
df['income_proxy'] = df['income_index'] * 0.95 + np.random.normal(0, 0.05, 100)


class HEDISPreprocessor:
    """
    A class to preprocess HEDIS data for compliance prediction.
    
    This pipeline performs the following steps:
    1. Initial data cleaning (dropping irrelevant columns).
    2. Filters data for the eligible population (denominator).
    3. Identifies and separates feature types (numerical vs. categorical).
    4. Removes highly correlated numerical features.
    5. Selects the optimal set of features using RFECV.
    6. Creates a scikit-learn preprocessing pipeline for imputation, scaling, and encoding.
    """
    def __init__(self, target_col='numercnt', denom_col='denomcnt'):
        self.target_col = target_col
        self.denom_col = denom_col
        self.preprocessor = None
        self.selected_features = None
        
        # Define columns based on the provided schema
        self.id_cols_to_drop = [
            'asdb_plankey', 'member_key', 'src_mbr_id', 'cust_no', 'cust_nm', 
            'indiv_anlytcs_id', 'iodb_mbr_key', 'iodb_plan_key', 'address_line_1', 
            'city', 'zip_code', 'bgfips'
        ]
        self.date_cols_to_drop = ['clssdt', 'bus_eff_dt', 'max_source_pstd_dts']
        
        # Note: social_risk_score is often a composite of other SDoH variables.
        # Including it with its components can introduce multicollinearity.
        # It's often best to use either the composite score or the individual features.
        self.other_cols_to_drop = ['social_risk_score']

    def _initial_cleanup(self, df, is_train=True):
        """Filters for denominator, drops unnecessary columns, and separates X and y."""
        # Filter for members in the denominator
        df_filtered = df[df[self.denom_col] == 1].copy()
        
        if is_train:
            y = df_filtered[self.target_col]
        else:
            y = None # No target in test data processing
        
        # Drop target and denominator columns from features
        cols_to_drop = [self.target_col, self.denom_col]
        X = df_filtered.drop(columns=cols_to_drop)
        
        # Drop irrelevant ID, date, and other specified columns
        all_cols_to_drop = self.id_cols_to_drop + self.date_cols_to_drop + self.other_cols_to_drop
        # Only drop columns that actually exist in the dataframe
        existing_cols_to_drop = [col for col in all_cols_to_drop if col in X.columns]
        X = X.drop(columns=existing_cols_to_drop)
        
        return X, y

    def _select_features(self, X, y):
        """Selects the best features using correlation analysis and RFECV."""
        print("Starting feature selection...")
        
        # 1. Identify numerical and categorical features
        numerical_features = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
        
        # 2. Remove highly correlated numerical features (helps Logistic Regression)
        corr_matrix = X[numerical_features].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.90)]
        X_reduced = X.drop(columns=to_drop)
        print(f"Dropped due to high correlation: {to_drop}")
        
        # Update feature lists
        numerical_features = [f for f in numerical_features if f not in to_drop]
        
        # 3. Use RFECV to find the optimal number of features
        # We need to temporarily preprocess the data for RFECV to run
        temp_num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median'))])
        temp_cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), 
                                  ('encoder', OneHotEncoder(handle_unknown='ignore'))])
        
        temp_preprocessor = ColumnTransformer(
            transformers=[
                ('num', temp_num_pipe, numerical_features),
                ('cat', temp_cat_pipe, categorical_features)
            ], remainder='passthrough'
        )
        
        # Use a fast model like XGBoost for feature selection
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        
        # The RFECV selector
        # min_features_to_select can be adjusted based on domain knowledge
        selector = RFECV(
            estimator=xgb, 
            step=1, 
            cv=StratifiedKFold(3), 
            scoring='roc_auc', 
            min_features_to_select=10
        )
        
        # Create a pipeline to run RFECV
        rfe_pipeline = Pipeline([
            ('preprocessor', temp_preprocessor),
            ('selector', selector)
        ])
        
        rfe_pipeline.fit(X_reduced, y)
        
        # Get the boolean mask of selected features and extract their names
        selected_features_mask = rfe_pipeline.named_steps['selector'].support_
        # Get feature names after one-hot encoding
        ohe_feature_names = rfe_pipeline.named_steps['preprocessor'].named_transformers_['cat'] \
                                     .named_steps['encoder'].get_feature_names_out(categorical_features)
        
        all_feature_names = numerical_features + list(ohe_feature_names)
        selected_feature_indices = np.where(selected_features_mask)[0]
        
        # The selector doesn't directly give us the original feature names. 
        # We get them from the original pre-transformed list.
        final_selected_features_mask = rfe_pipeline.named_steps['selector'].get_support()
        original_features = numerical_features + categorical_features
        self.selected_features = [feat for feat, selected in zip(original_features, final_selected_features_mask) if selected]
        
        print(f"RFECV selected {len(self.selected_features)} features: {self.selected_features}")
        
        return self.selected_features

    def fit_transform(self, df):
        """Fits the preprocessor and transforms the training data."""
        print("Fitting and transforming data...")
        X, y = self._initial_cleanup(df)
        
        # Run feature selection to determine the best features
        self.selected_features = self._select_features(X, y)
        
        # Refine numerical and categorical lists based on selection
        numerical_features = X[self.selected_features].select_dtypes(include=np.number).columns.tolist()
        categorical_features = X[self.selected_features].select_dtypes(exclude=np.number).columns.tolist()
        
        # Create the final preprocessing pipelines
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # Create the ColumnTransformer to apply different transformations to different columns
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough' # Keep other selected columns if any (should be none)
        )
        
        X_processed = self.preprocessor.fit_transform(X[self.selected_features])
        print("Preprocessing complete.")
        return X_processed, y

    def transform(self, df):
        """Transforms new data using the fitted preprocessor."""
        if self.preprocessor is None or self.selected_features is None:
            raise RuntimeError("You must fit the preprocessor before transforming data.")
        
        print("Transforming new data...")
        X, _ = self._initial_cleanup(df, is_train=False)
        X_processed = self.preprocessor.transform(X[self.selected_features])
        print("Transformation complete.")
        return X_processed

# --- Main execution block to demonstrate usage ---
if __name__ == '__main__':
    # 1. Instantiate the preprocessor
    preprocessor = HEDISPreprocessor(target_col='numercnt', denom_col='denomcnt')
    
    # 2. Split data into training and testing sets
    X_raw = df.drop(columns=['numercnt'])
    y_raw = df['numercnt']
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        df, df['numercnt'], test_size=0.25, random_state=42, stratify=df['numercnt']
    )
    
    # 3. Fit the preprocessor on the training data and transform it
    X_train_processed, y_train = preprocessor.fit_transform(X_train_raw)
    
    # 4. Transform the test data using the same fitted preprocessor
    X_test_processed = preprocessor.transform(X_test_raw)
    
    print(f"\nShape of processed training features: {X_train_processed.shape}")
    print(f"Shape of processed test features: {X_test_processed.shape}")
    
    # 5. Build final modeling pipelines with SMOTE for class imbalance
    # This ensures SMOTE is only applied to the training data during model fitting
    
    # Logistic Regression Pipeline
    lr_pipeline = ImbPipeline(steps=[
        ('smote', SMOTE(random_state=42)),
        ('classifier', LogisticRegression(random_state=42, solver='liblinear'))
    ])

    # XGBoost Pipeline
    xgb_pipeline = ImbPipeline(steps=[
        ('smote', SMOTE(random_state=42)),
        ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
    ])

    # 6. Train the models
    print("\nTraining Logistic Regression model...")
    lr_pipeline.fit(X_train_processed, y_train)
    
    print("Training XGBoost model...")
    xgb_pipeline.fit(X_train_processed, y_train)
    
    print("\nModels trained successfully.")
    
    # You can now use these fitted pipelines to make predictions
    # lr_predictions = lr_pipeline.predict(X_test_processed)
    # xgb_predictions = xgb_pipeline.predict(X_test_processed)
    
    # Example: Check accuracy
    lr_score = lr_pipeline.score(X_test_processed, y_test_raw)
    xgb_score = xgb_pipeline.score(X_test_processed, y_test_raw)
    
    print(f"\nLogistic Regression Test Accuracy: {lr_score:.4f}")
    print(f"XGBoost Test Accuracy: {xgb_score:.4f}")