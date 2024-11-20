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
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())

############################################################################################################
# Data Splitter

class DataSplitter:
    """
    DataSplitter class to prepare data for forecasting models
     N = Sequence length 
     M = Forecast horizon
    """
    def __init__(self, df, target_col, N, M, train_cutoff_date=None):
        self.df = df.copy()
        self.target_col = target_col
        self.N = N
        self.M = M
        self.train_cutoff_date = train_cutoff_date or self._default_cutoff_date()
        self.scaler = MinMaxScaler(feature_range=(1, 2))  # Single scaler for all data
        
    def _default_cutoff_date(self):
        total_days = (self.df['date'].max() - self.df['date'].min()).days
        cutoff = self.df['date'].min() + timedelta(days=int(total_days * 0.8))
        return cutoff.strftime('%Y-%m-%d')
    
    def _prepare_data(self):
        df = self.df[['date', self.target_col]].copy()
        df = df.rename(columns={'date': 'ds', self.target_col: 'y'})
        df = df.sort_values('ds')
        
        # Handle missing values
        df['y'] = df['y'].interpolate(method='linear')
        df['y'] = df['y'].fillna(method='ffill').fillna(method='bfill')
        
        # Scale all the data at once
        df['y'] = self.scaler.fit_transform(df['y'].values.reshape(-1, 1))
        
        return df
    
    def _create_sequences(self, data, is_training=True):
        sequences = []
        end_idx = len(data) - self.M if is_training else len(data)
            
        for i in range(self.N, end_idx - self.M + 1):
            sequence = data.iloc[i-self.N:i+self.M].copy()
            sequences.append(sequence)
            
        return sequences
    
    def split_data(self):
        df = self._prepare_data()
        train_data = df[df['ds'] <= self.train_cutoff_date].copy()
        test_data = df[df['ds'] > self.train_cutoff_date].copy()
        
        return (self._create_sequences(train_data, is_training=True),
                self._create_sequences(test_data, is_training=False))
    
    def inverse_transform(self, values):
        """Helper method to inverse transform scaled values"""
        return self.scaler.inverse_transform(values)


############################################################################################################
# Custom Mirror Model

class CustomMirrorModel:
    """
    A custom forecasting model that combines historical patterns with current trends
    to predict future measure rates while maintaining reasonable proximity to prior year values.
    """
    def __init__(self):
        self.train_data = None
        self.fitted = False
        self.prior_year_data = None
        self.current_year_data = None
        self.trend_coefficient = None
        self.seasonal_patterns = None
        
    def _prepare_data(self, df):
        """
        Prepare and organize data by measurement year
        """
        # Convert cycle_id to year and period
        df['year'] = df['cycle_id'].str.split('-').str[0].astype(int)
        df['period'] = df['cycle_id'].str.split('-').str[1].astype(int)
        
        # Get current and prior year
        current_year = df['year'].max()
        
        # Separate current and prior year data
        self.current_year_data = df[df['year'] == current_year].copy()
        self.prior_year_data = df[df['year'] == current_year - 1].copy()
        
        return current_year
        
    def _calculate_trend_coefficient(self):
        """
        Calculate trend coefficient based on comparison of current vs prior year rates
        with enhanced dampening for stability
        """
        # Get overlapping periods
        max_current_period = self.current_year_data['period'].max()
        
        # Compare rates for overlapping periods
        current_rates = self.current_year_data[self.current_year_data['period'] <= max_current_period]['measure_rate']
        prior_rates = self.prior_year_data[self.prior_year_data['period'] <= max_current_period]['measure_rate']
        
        # Calculate average difference
        avg_diff = (current_rates.mean() - prior_rates.mean()) / prior_rates.mean()
        
        # Allow slightly more trend influence
        # Increased from ±0.05 to ±0.065 for less conservative projections
        self.trend_coefficient = np.clip(avg_diff, -0.065, 0.065)
        
    def _calculate_seasonal_patterns(self):
        """
        Calculate seasonal patterns from prior year data
        """
        # Calculate seasonal factors from prior year
        self.seasonal_patterns = self.prior_year_data.set_index('period')['measure_rate']
        
    def fit(self, df):
        """
        Fit the model to the training data
        """
        self.train_data = df.copy()
        current_year = self._prepare_data(df)
        
        # Calculate trend and seasonal patterns
        self._calculate_trend_coefficient()
        self._calculate_seasonal_patterns()
        
        self.fitted = True
        return self
        
    def predict(self, steps):
        """
        Generate predictions for the specified number of steps ahead
        with enhanced dampening for final periods
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        # Get last known period
        last_period = self.current_year_data['period'].max()
        
        # Generate predictions
        predictions = []
        for i in range(steps):
            target_period = last_period + i + 1
            
            if target_period > 16:  # Don't predict beyond period 16
                break
                
            # Get prior year value for this period
            prior_value = self.seasonal_patterns[target_period]
            
            # Calculate progressive dampening factor
            # Increases dampening as we get closer to period 16
            steps_to_end = 16 - target_period
            dampening_factor = max(0.4, steps_to_end / 16)  # Minimum 0.4 dampening (increased from 0.3)
            
            # Apply progressively dampened trend adjustment
            adjusted_trend = self.trend_coefficient * dampening_factor
            predicted_value = prior_value * (1 + adjusted_trend)
            
            # Apply additional dampening for final periods (13-16)
            if target_period >= 13:
                # Calculate how close the prediction should stay to prior year value
                proximity_factor = 1 - ((target_period - 12) / 4) * 0.6  # Reduced from 0.7 to 0.6 for less dampening
                predicted_value = (predicted_value * proximity_factor + 
                                 prior_value * (1 - proximity_factor))
            
            # Apply smoothing for transition from last known value
            if i == 0 and last_period < 16:
                last_known_value = self.current_year_data[
                    self.current_year_data['period'] == last_period
                ]['measure_rate'].iloc[0]
                # Enhanced smoothing for transition
                alpha = 0.6  # Reduced from 0.7 for smoother transition
                predicted_value = (alpha * predicted_value + 
                                 (1 - alpha) * last_known_value)
            
            predictions.append(predicted_value)
            
        return np.array(predictions)
        
    def calculate_forecast_horizon(self, cycle_id):
        """
        Calculate the forecast horizon based on the max cycle_id
        """
        year, period = map(int, cycle_id.split('-'))
        return 16 - period  # Calculate remaining periods until YYYY-16
    
############################################################################################################
# Exponential Smoothing Model - Holt-Winters - Numerator-Denominator

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
        
    def _create_calendar_date(self, row):
        """Convert cycle_id to calendar date."""
        year = int(row['measurement_year'])
        cycle = int(row['cycle_id'].split('-')[1])
        month = ((cycle - 1) % 12) + 1
        return pd.Timestamp(year=year, month=month, day=1)
        
    def prepare_data(self, df):
        """Prepare time series data for forecasting."""
        df = df.copy()
        
        # Create calendar dates if not present
        if 'calendar_date' not in df.columns:
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
        Expects a DataFrame with columns: num, den, measurement_year, cycle_id
        """
        prepared_data = self.prepare_data(data)
        
        # Store current year and last date for forecasting
        self.current_year = str(prepared_data['measurement_year'].max())
        self.last_train_date = prepared_data.index[-1]
        
        # Create time series with proper frequency
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
    


############################################################################################################
# Model Evaluator

class ModelEvaluator:
    """
    Evaluates the performance of time series forecasting models
    """
    def __init__(self):
        self.metrics = {}
        
    def evaluate(self, model, df, test_data=None, run_type="test"):
        """
        Evaluate model performance
        """
        if run_type == "test" and test_data is not None:
            predictions = model.predict(len(test_data))
            rmse = np.sqrt(mean_squared_error(test_data, predictions))
            mae = mean_absolute_error(test_data, predictions)
            
            self.metrics = {
                'rmse': rmse,
                'mae': mae,
                'predictions': predictions
            }
        
        elif run_type == "prod":
            last_cycle = df['cycle_id'].iloc[-1]
            forecast_horizon = model.calculate_forecast_horizon(last_cycle)
            predictions = model.predict(forecast_horizon)
            
            self.metrics = {
                'predictions': predictions,
                'forecast_horizon': forecast_horizon
            }
        
        return self.metrics
    
    def plot_results(self, df, predictions, run_type="test", test_data=None,
                    title="Forecast vs Actual", model_type=""):
        """
        Plot the forecasting results with enhanced styling and model identification
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe containing the data
        predictions : array-like
            Predicted values
        run_type : str
            Either "test" or "prod"
        test_data : array-like, optional
            Test data for validation
        title : str
            Base title for the plot
        model_type : str
            Type of model used (e.g., 'sarima', 'hw_damped')
        """
        # Set style
        plt.style.use('fivethirtyeight')
        
        # Create figure and axis objects with a larger figure size
        fig, ax = plt.subplots(figsize=(15, 8), dpi=100)
        
        # Define a modern color palette
        colors = {
            'train': '#2C3E50',  # Dark blue-gray
            'test': '#27AE60',   # Green
            'pred': '#E74C3C',   # Red
            'grid': '#ECF0F1'    # Light gray
        }
        
        # Plot data
        if run_type == "test" and test_data is not None:
            train_data = df['measure_rate'].iloc[:-len(test_data)].values
            
            # Plot training data with line and dots
            train_indices = range(len(train_data))
            ax.plot(train_indices, train_data,
                    label='Training Data', 
                    color=colors['train'],
                    linewidth=2)
            ax.scatter(train_indices, train_data,
                    color=colors['train'],
                    s=30,
                    alpha=0.6)
            
            # Plot test data and predictions with dots
            test_idx = range(len(train_data), len(train_data) + len(test_data))
            ax.plot(test_idx, test_data, 
                    label='Test Data',
                    color=colors['test'],
                    linewidth=2)
            ax.scatter(test_idx, test_data,
                    color=colors['test'],
                    s=30,
                    alpha=0.6)
            
            ax.plot(test_idx, predictions,
                    label='Predictions',
                    color=colors['pred'],
                    linestyle='--',
                    linewidth=2)
            ax.scatter(test_idx, predictions,
                    color=colors['pred'],
                    s=30,
                    alpha=0.6,
                    marker='x')
            
        else:  # prod mode
            train_data = df['measure_rate'].values
            
            # Plot historical data with line and dots
            train_indices = range(len(train_data))
            ax.plot(train_indices, train_data,
                    label='Historical Data',
                    color=colors['train'],
                    linewidth=2)
            ax.scatter(train_indices, train_data,
                    color=colors['train'],
                    s=30,
                    alpha=0.6)
            
            # Plot predictions with dots
            pred_idx = range(len(train_data), len(train_data) + len(predictions))
            ax.plot(pred_idx, predictions,
                    label='Forecast',
                    color=colors['pred'],
                    linestyle='--',
                    linewidth=2)
            ax.scatter(pred_idx, predictions,
                    color=colors['pred'],
                    s=30,
                    alpha=0.6,
                    marker='x')
        
        # Format the title with model type
        model_name = f"{model_type.upper()} Model" if model_type else ""
        if 'measure_id_key' in df.columns and 'book_name' in df.columns:
            title = f"{title}\n{df['measure_id_key'].iloc[0]} ({df['book_name'].iloc[0]})"
            if model_name:
                title = f"{title}\n{model_name}"
        elif model_name:
            title = f"{title}\n{model_name}"
        
        # Set title and labels with enhanced font
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Time Period', fontsize=12, fontweight='bold')
        ax.set_ylabel('Measure Rate', fontsize=12, fontweight='bold')
        
        # Customize grid
        ax.grid(True, linestyle='--', alpha=0.7, color=colors['grid'])
        ax.set_axisbelow(True)  # Place grid lines behind the data
        
        # Customize legend
        ax.legend(loc='upper left', 
                frameon=True,
                fancybox=True,
                shadow=True,
                fontsize=10)
        
        # Customize spines
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        
        # Adjust layout
        plt.tight_layout()
        
        # Create identifier including model type
        if 'measure_id_key' in df.columns and 'book_name' in df.columns:
            identifier = f"{df['measure_id_key'].iloc[0]}_{df['book_name'].iloc[0]}"
        else:
            identifier = "general"
        
        # Save figure with model type in filename
        if not os.path.exists(f'output/img/{run_type}'):
            os.makedirs(f'output/img/{run_type}')
        model_suffix = f"_{model_type}" if model_type else ""
        plt.savefig(f'output/img/{run_type}/forecast_results_{identifier}{model_suffix}_{run_type}.png',
                    bbox_inches='tight',
                    dpi=300)
        plt.close()
        
    def create_results_df(self, df, predictions, run_type="test", test_data=None):
        """
        Create a results DataFrame
        """
        # Create a copy and reset the index to keep the date as a column
        results_df = df.copy()
        if isinstance(results_df.index, pd.DatetimeIndex):
            results_df = results_df.reset_index()
        
        if run_type == "test" and test_data is not None:
            # Add predictions only for test period
            results_df.iloc[-len(test_data):, results_df.columns.get_loc('predictions')] = predictions
            
            # Add metrics
            results_df['rmse'] = self.metrics.get('rmse', None)
            results_df['mae'] = self.metrics.get('mae', None)
        
        else:  # prod mode
            last_cycle = results_df['cycle_id'].iloc[-1]
            year = int(last_cycle.split('-')[0])
            last_period = int(last_cycle.split('-')[1])
            
            # Create future cycle_ids
            future_cycles = [
                f"{year}-{str(i).zfill(2)}" 
                for i in range(last_period + 1, last_period + len(predictions) + 1)
            ]
            
            # Create future dates
            last_date = results_df['date'].iloc[-1]
            future_dates = pd.date_range(
                start=last_date + timedelta(days=30),
                periods=len(predictions),
                freq='30D'
            )
            
            # Create forecast rows
            forecast_rows = []
            for i in range(len(predictions)):
                new_row = results_df.iloc[-1].copy()
                new_row['date'] = future_dates[i]
                new_row['cycle_id'] = future_cycles[i]
                new_row['measure_rate'] = None
                new_row['predictions'] = predictions[i]
                forecast_rows.append(new_row)
            
            # Append forecast rows
            forecast_df = pd.DataFrame(forecast_rows)
            results_df = pd.concat([results_df, forecast_df], ignore_index=True)
        
        # Set the date as index again
        results_df.set_index('date', inplace=True)
        return results_df
    

############################################################################################################
# Utility function to run the entire prediction process

def make_predictions(df, run_type="test", test_size=7, model_id='hw_damped'):
    """
    Utility function to run the entire prediction process
    Now selects model_class based on model_id parameter
    """
    if 'predictions' not in df.columns:
        df['predictions'] = None

    # Map model_id to model_class
    model_classes = {
        'hw_damped': '',
        'sarima': '',
        'custom_mirror': CustomMirrorModel,
        'hw_nd': HWNDModel,  # Holt-Winters Numerator-Denominator
    }

    if model_id not in model_classes:
        raise ValueError(f"Model type {model_id} not recognized")

    model_class = model_classes[model_id]

    # Instantiate the model
    if model_id == 'hw_damped':
        model = model_class(seasonal_periods=16)
    else:
        model = model_class()

    evaluator = ModelEvaluator()

    if run_type == "test":
        train_df = df.iloc[:-test_size]
        test_data = df['measure_rate'].iloc[-test_size:].values

        # For models that require the entire DataFrame
        if model_id == 'custom_mirror':
            model.fit(train_df.reset_index())
        else:
            model.fit(train_df['measure_rate'].values)

        metrics = evaluator.evaluate(model, df, test_data, run_type="test")

        results_df = evaluator.create_results_df(
            df,
            metrics['predictions'],
            run_type="test",
            test_data=test_data
        )

        evaluator.plot_results(
            df,
            metrics['predictions'],
            run_type="test",
            test_data=test_data,
            model_type=model_id
        )

    else:  # prod mode
        # For models that require the entire DataFrame
        if model_id == 'custom_mirror':
            model.fit(df.reset_index())
        else:
            model.fit(df['measure_rate'].values)

        metrics = evaluator.evaluate(model, df, run_type="prod")

        results_df = evaluator.create_results_df(
            df,
            metrics['predictions'],
            run_type="prod"
        )

        evaluator.plot_results(
            df,
            metrics['predictions'],
            run_type="prod",
            model_type=model_id
        )

    return results_df, model, evaluator