import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import timedelta
import os

class HWNDModel:
    """
    Holts-Winters Numerator-Denominator (HWND) Model
    Forecasts numerator and denominator separately using Holt-Winters models
    and combines them to predict the measure rate.
    """
    def __init__(self, seasonal_periods=16):
        self.seasonal_periods = seasonal_periods
        self.num_model = None
        self.den_model = None
        self.last_train_date = None
        self.current_year = None
        self.forecast_horizon = None
        self._original_df = None  # Store the original DataFrame
        
    def _create_calendar_date(self, row):
        """Convert cycle_id to calendar date."""
        year = int(row['measurement_year'])
        cycle = int(row['cycle_id'].split('-')[1])
        month = ((cycle - 1) % 12) + 1
        return pd.Timestamp(year=year, month=month, day=1)
        
    def prepare_data(self, df):
        """Prepare time series data for forecasting."""
        df = df.copy()
        df['calendar_date'] = df.apply(self._create_calendar_date, axis=1)
        df = df.sort_values('calendar_date')
        
        # Ensure dates have frequency
        df = df.set_index('calendar_date')
        df.index = pd.DatetimeIndex(df.index).to_period('M')
        
        return df
    
    def calculate_forecast_horizon(self, last_cycle):
        """Calculate number of periods to forecast."""
        year, cycle = map(int, last_cycle.split('-'))
        return 16 - cycle
        
    def fit(self, data):
        """
        Fit the model to training data.
        Can accept either:
        1. A DataFrame with columns: num, den, measurement_year, cycle_id
        2. A numpy array of measure_rate values (from make_predictions)
        """
        # Check if input is numpy array (from make_predictions)
        if isinstance(data, np.ndarray):
            if not hasattr(self, '_original_df'):
                raise ValueError("Model must be initialized with original DataFrame first")
            
            # Create synthetic num/den that would give these measure rates
            # Using 100 as base denominator for simplicity
            base_den = 100
            num_values = data * base_den
            den_values = np.full_like(num_values, base_den)
            
            # Create time series
            dates = pd.date_range(start='2023-01-01', periods=len(data), freq='M')
            num_series = pd.Series(num_values, index=dates)
            den_series = pd.Series(den_values, index=dates)
            
        else:  # Input is DataFrame
            self._original_df = data.copy()  # Store original DataFrame
            prepared_data = self.prepare_data(data)
            
            # Store current year and last date for forecasting
            self.current_year = str(prepared_data['measurement_year'].max())
            self.last_train_date = prepared_data.index[-1]
            
            # Create time series
            num_series = prepared_data['num'].asfreq('M')
            den_series = prepared_data['den'].asfreq('M')
        
        # Fit numerator model
        self.num_model = ExponentialSmoothing(
            num_series,
            seasonal_periods=12,
            trend='add',
            seasonal='add',
            initialization_method='estimated'
        ).fit()
        
        # Fit denominator model
        self.den_model = ExponentialSmoothing(
            den_series,
            seasonal_periods=12,
            trend='add',
            seasonal='add',
            initialization_method='estimated'
        ).fit()
        
        return self
        
    def predict(self, forecast_horizon):
        """
        Generate predictions for specified number of periods.
        Returns measure_rate predictions.
        """
        if not (self.num_model and self.den_model):
            raise ValueError("Models must be trained before predicting")
            
        # Generate forecasts
        num_forecast = self.num_model.forecast(forecast_horizon)
        den_forecast = self.den_model.forecast(forecast_horizon)
        
        # Calculate measure rate predictions
        measure_rate_predictions = (
            np.round(num_forecast) / np.round(den_forecast)
        ).values
        
        return measure_rate_predictions
    
    def get_forecast_dataframe(self, original_df, predictions):
        """
        Create a forecast DataFrame with all required columns.
        """
        last_cycle = original_df['cycle_id'].iloc[-1]
        year, cycle = map(int, last_cycle.split('-'))
        
        # Generate future cycles
        future_cycles = [
            f"{year}-{str(i).zfill(2)}" 
            for i in range(cycle + 1, cycle + len(predictions) + 1)
        ]
        
        # Generate future dates
        future_dates = pd.period_range(
            start=self.last_train_date + 1,
            periods=len(predictions),
            freq='M'
        )
        
        # Generate numerator and denominator predictions
        num_predictions = np.round(self.num_model.forecast(len(predictions))).astype(int)
        den_predictions = np.round(self.den_model.forecast(len(predictions))).astype(int)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'book_name': original_df['book_name'].iloc[0],
            'measure_id_key': original_df['measure_id_key'].iloc[0],
            'measurement_year': year,
            'cycle_id': future_cycles,
            'calendar_date': future_dates.astype(str),
            'num': num_predictions,
            'den': den_predictions,
            'measure_rate': predictions,
            'predictions': predictions  # Required by ModelEvaluator
        })
        
        return forecast_df
    
    def save_forecast(self, forecast_df, model_id, output_dir='output/forecast'):
        """Save forecast results to CSV."""
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"{forecast_df['book_name'].iloc[0]}_{forecast_df['measure_id_key'].iloc[0]}_{model_id}.csv"
        filepath = os.path.join(output_dir, filename)
        
        forecast_df.to_csv(filepath, index=False)
        return filepath

# Usage example:
"""
# First, initialize with complete DataFrame to set up the model
df = pd.read_csv('your_data.csv')
model = HWNDModel(seasonal_periods=16)
model._original_df = df  # Store the original DataFrame

# Now you can use it with make_predictions
results_df, model, evaluator = make_predictions(
    df, 
    run_type="test",
    test_size=7,
    model_id='hwnd'
)
"""