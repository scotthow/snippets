import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime
import os

class MeasureRateForecaster:
    def __init__(self, seasonal_periods=16):
        self.seasonal_periods = seasonal_periods
        self.num_model = None
        self.den_model = None
        
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
        
        # Calculate forecast horizon
        current_year = df['measurement_year'].max()
        current_year_data = df[df['measurement_year'] == current_year]
        max_cycle = int(current_year_data['cycle_id'].iloc[-1].split('-')[1])
        self.forecast_horizon = 16 - max_cycle
        
        return df
        
    def train(self, df):
        """Train Holt-Winters models for numerator and denominator."""
        prepared_data = self.prepare_data(df)
        
        # Create time series with proper frequency
        num_series = prepared_data['num'].asfreq('M')
        den_series = prepared_data['den'].asfreq('M')
        
        self.num_model = ExponentialSmoothing(
            num_series,
            seasonal_periods=12,
            trend='add',
            seasonal='add',
            initialization_method='estimated'
        ).fit()
        
        self.den_model = ExponentialSmoothing(
            den_series,
            seasonal_periods=12,
            trend='add',
            seasonal='add',
            initialization_method='estimated'
        ).fit()
        
        return self
        
    def forecast(self, df):
        """Generate forecasts for the remaining cycles in current year."""
        if not (self.num_model and self.den_model):
            raise ValueError("Models must be trained before forecasting")
            
        prepared_data = self.prepare_data(df)
        last_date = prepared_data.index[-1]
        
        # Generate future dates with proper frequency
        future_dates = pd.period_range(
            start=last_date + 1,
            periods=self.forecast_horizon,
            freq='M'
        )
        
        # Generate forecasts
        num_forecast = self.num_model.forecast(self.forecast_horizon)
        den_forecast = self.den_model.forecast(self.forecast_horizon)
        
        # Get the last cycle number and create next cycle IDs
        current_year = str(prepared_data['measurement_year'].max())
        last_cycle = int(prepared_data[prepared_data['measurement_year'] == int(current_year)]['cycle_id'].iloc[-1].split('-')[1])
        
        forecast_cycles = []
        for i in range(1, self.forecast_horizon + 1):
            next_cycle = last_cycle + i
            forecast_cycles.append(f"{current_year}-{str(next_cycle).zfill(2)}")
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'book_name': df['book_name'].iloc[0],
            'measure_id_key': df['measure_id_key'].iloc[0],
            'measurement_year': current_year,
            'cycle_id': forecast_cycles,
            'predicted_num': np.round(num_forecast).astype(int),
            'predicted_den': np.round(den_forecast).astype(int),
            'calendar_date': future_dates.astype(str)
        })
        
        # Calculate forecasted measure rate
        forecast_df['forecasted_measure_rate'] = (
            forecast_df['predicted_num'] / forecast_df['predicted_den']
        )
        
        return forecast_df
        
    def save_forecast(self, forecast_df, model_id, output_dir='output/forecast'):
        """Save forecast results to CSV."""
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"{forecast_df['book_name'].iloc[0]}_{forecast_df['measure_id_key'].iloc[0]}_{model_id}.csv"
        filepath = os.path.join(output_dir, filename)
        
        forecast_df.to_csv(filepath, index=False)
        return filepath

def run_forecast(df, model_id):
    """Run the complete forecasting process."""
    forecaster = MeasureRateForecaster()
    forecaster.train(df)
    forecast_results = forecaster.forecast(df)
    filepath = forecaster.save_forecast(forecast_results, model_id)
    return forecast_results