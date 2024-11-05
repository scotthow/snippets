# -*- coding: utf-8 -*-
"""
"""

# !pip install plotly kaleido statsmodels scikit-learn pandas numpy

import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose
import itertools
import statsmodels.api as sm
from scipy import stats
from datetime import datetime
import os, sys
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from typing import Dict, Tuple, Any, Optional
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings
import seaborn as sns
import logging

# Add BASE_DIR to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from app.utils import pre_processing, data_finder
from app.models.predict import *
from app.config import Config

fc_list = Config.FC_LIST
test_size = Config.TEST_SIZE
MEASUREMENT_YEAR = Config.MEASUREMENT_YEAR

TODAY = datetime.today().strftime('%Y-%m-%d')

############################################################################################################
# Forecasting class

class TimeSeriesForecaster:
    def __init__(self, df, model_type='hw-damped'):
        """
        df : pandas.DataFrame
        df columns are as follows: 
            measurement_year          int64
            book_name                object
            hybrid_measure_flag     float64
            inverse_measure_flag    float64
            measure_id_key           object
            measure_rate            float64
            cycle_id                 object
        """
        self.df = df.copy().reset_index(drop=True)
        self.model_type = model_type
        self.model = None
        
    def preprocess_data(self):
        """
        Preprocess the data by creating proper datetime index
        """
        base_date = datetime(2022, 1, 1)
        date_map = {}
        current_date = base_date

        years = [str(year) for year in self.df['cycle_id'].str[:4].unique()]

        for year in years:
            for month in range(1, 17):
                cycle_id = f"{year}-{str(month).zfill(2)}"
                date_map[cycle_id] = current_date
                current_date = current_date + timedelta(days=30)

        processed_df = self.df.copy()
        processed_df['date'] = processed_df['cycle_id'].map(date_map)
        processed_df = processed_df.reset_index(drop=True)
        processed_df = processed_df.sort_values('date')
        processed_df.set_index('date', inplace=True)

        return processed_df

    def make_predictions(self, run_type="test", test_size=7):
        """
        Make predictions using the specified model type
        """
        self.df = self.df.copy().reset_index(drop=True)
        processed_df = self.preprocess_data()
        
        if self.model_type == 'hw-damped':
            results_df, self.model, evaluator = make_predictions(
                processed_df, 
                run_type=run_type,
                test_size=test_size,
                model_class=HoltWintersModel_V2
            )
            return results_df
            
        elif self.model_type == 'sarima':
            results_df, self.model, evaluator = make_predictions(
                processed_df, 
                run_type=run_type,
                test_size=test_size,
                model_class=SARIMAModel_V2
            )
            return results_df
            
        else:
            raise ValueError(f"Model type {self.model_type} not recognized")

    def create_test_set(self, test_size=7):
        """
        Split data into training and test sets
        """
        processed_df = self.preprocess_data()
        train_data = processed_df.iloc[:-test_size].copy()
        test_data = processed_df.iloc[-test_size:].copy()
        return train_data, test_data


############################################################################################################
# Main function
        
def main():

    # Generate Data
    dg = data_finder.DataGenerator()
    dg.save_to_csv()
    
    # Load data
    _df = pd.read_csv('output/df_sample.csv')
    _df['measure_id_key'] = _df['measure_id_key'].astype(str)

    # test_size = 7

    df_list = []
    # Initialize empty dataframe
    cols = [
        'measurement_year',
        'book_name',
        'market_name',
        'measure_id_key',
        'internal_qpr_measure_key',
        'measure_tla',
        'measure_desc',
        'measure_desc_long',
        'msr_rate_char',
        'instance_data_through_dt',
        'rollup_flag',
        'hybrid_measure_flag',
        'inverse_measure_flag',
        'forecasted_flag',
        'lower_rate_is_better_flag',
        'age_span_cd',
        'gender_cd',
        'org_id',
        'sub_id',
        'synthetic',
        'cycle_id',
        'epop_count',
        'den',
        'num',
        'exclusion_count',
        'measure_rate'
    ]

    df_combined = pd.DataFrame(columns=cols)

    forecast_cols = [
        'measurement_year', 'book_name', 'measure_id_key', 'cycle_id', 'measure_rate', 'rmse', 'mae', 'model_id', 'time_series_date', 'pred_date', 'pred_flag'
    ]

    df_forecast_combined = pd.DataFrame(columns=forecast_cols)

    for population, measure, model_id in fc_list:
        print(f"\nProcessing forecast for Population: {population}, Measure: {measure}")

        # Filter and prepare data
        df = _df[(_df['book_name'] == population) & (_df['measure_id_key'] == measure)].copy()
        df['measure_rate'] = df['measure_rate'].round(4)
        df['measure_rate'] = df['measure_rate'].apply(lambda x: float(f"{x:.4f}"))
        df.sort_values(['cycle_id'], inplace=True)
        # Save to CSV - for testing
        df.to_csv(f'output/df_{population}_{measure}_pre_synthetic_step.csv', index=False)

        # Inverse measure adjuster
        # If df is not an inverse measure, just returns the original df
        inverse_rate_adjuster = pre_processing.InverseMeasureRateAdjuster(df)
        df = inverse_rate_adjuster.adjust()
        # Save to CSV - for testing
        df.to_csv(f'output/df_{population}_{measure}_post_inverse_adjustment.csv', index=False)

        # Add synthetic rows
        synthetic_creator = pre_processing.SyntheticRowsCreator(df)
        synthetic_creator.preprocess_2022_measure_rates()
        df = synthetic_creator.get_processed_data()

        #################################################
        # Extract values for the final forecast dataframe
        curr_cycle_id = df['cycle_id'].max()
        curr_rate = df[df['cycle_id'] == curr_cycle_id]['measure_rate'].values[0]

        my22_last_admin_cycle_id = f"2022-15"
        my22_last_admin_rate = df[df['cycle_id'] == my22_last_admin_cycle_id]['measure_rate'].values[0]
        my22_reported_cycle_id = f"2022-16"
        my22_reported_rate = df[df['cycle_id'] == my22_reported_cycle_id]['measure_rate'].values[0]
        print(f"Current cycle_id: {curr_cycle_id}")
        print(f"Current rate: {curr_rate}")
        print(f"Last admin cycle_id (2022): {my22_last_admin_cycle_id}")
        print(f"Last admin rate (2022): {my22_last_admin_rate}")

        my23_last_admin_cycle_id = f"2023-15"
        my23_last_admin_rate = df[df['cycle_id'] == my23_last_admin_cycle_id]['measure_rate'].values[0]
        my23_reported_cycle_id = f"2023-16"
        my23_reported_rate = df[df['cycle_id'] == my23_reported_cycle_id]['measure_rate'].values[0]
        print(f"Last admin cycle_id (2023): {my23_last_admin_cycle_id}")
        print(f"Last admin rate (2023): {my23_last_admin_rate}")
        print(f"Reported cycle_id (2023): {my23_reported_cycle_id}")
        print(f"Reported rate (2023): {my23_reported_rate}")

        num_data_points = len(df)
        num_data_points_synth = len(df[df['synthetic'] == 1])
        print(f"Number of data points: {num_data_points}")
        print(f"Number of data points (synthetic): {num_data_points_synth}")

        # Combine dataframes
        df_combined = pd.concat([df_combined, df], ignore_index=True)
        df_combined.to_csv('output/df_combined.csv', index=False)

        # Prepare forecast columns
        fc_cols = [
            'book_name', 'measure_id_key', 'measure_rate', 'cycle_id'
        ]
        df = df[fc_cols]
        
        # Create forecaster instances and generate predictions
        if model_id in ['hw-damped', 'sarima']:
            test_forecaster = TimeSeriesForecaster(df, model_type=model_id)
            prod_forecaster = TimeSeriesForecaster(df, model_type=model_id)
            
            # Run test mode
            results_df_test = test_forecaster.make_predictions(run_type="test")
            # # Keep only rows in test period
            # results_df_test = results_df_test[-test_size:] 
            # # Save results
            # results_df_test.to_csv(
            #     f'output/forecasts/results_{population}_{measure}_{model_id}_test_{TODAY}.csv',
            #     index=True
            # )
            # Take the value of rmse and mae in the last row
            rmse = results_df_test['rmse'].iloc[-1]
            mae = results_df_test['mae'].iloc[-1]
            
            # Run prod mode
            results_df_prod = prod_forecaster.make_predictions(run_type="prod")
            # Keep only rows with predictions
            results_df_prod = results_df_prod[results_df_prod['predictions'].notnull()]
            # Drop measure_rate column; this only applied to historical data that we just excluded
            # measure_rate will be replaced by predictions
            results_df_prod.drop(columns=['measure_rate'], inplace=True)
            # Add rows
            results_df_prod['measurement_year'] = MEASUREMENT_YEAR
            results_df_prod['rmse'] = round(rmse, 3)
            results_df_prod['mae'] = round(mae, 3)
            results_df_prod['model_id'] = model_id
            results_df_prod['time_series_date'] = results_df_prod.index
            results_df_prod['pred_date'] = TODAY
            results_df_prod['pred_flag'] = 1
            # Rename columns
            results_df_prod.rename(columns={'predictions': 'measure_rate'}, inplace=True)
            # Add previously extracted values
            results_df_prod['my22_last_admin_rate'] = my22_last_admin_rate
            results_df_prod['my22_reported_rate'] = my22_reported_rate
            results_df_prod['my23_last_admin_rate'] = my23_last_admin_rate
            results_df_prod['my23_reported_rate'] = my23_reported_rate
            results_df_prod['num_data_points'] = num_data_points
            results_df_prod['num_data_points_synth'] = num_data_points_synth

            # results_df_prod.to_csv(
            #     f'output/forecasts/results_{population}_{measure}_{model_id}_prod_{TODAY}.csv',
            #     index=True
            # )

            print(list(results_df_prod.columns))

            # Combine dataframes
            df_forecast_combined = pd.concat([df_forecast_combined, results_df_prod], ignore_index=True)
            df_forecast_combined.to_csv(f'output/df_forecast_combined_prod_{TODAY}.csv', index=False)

if __name__ == '__main__':
    main()