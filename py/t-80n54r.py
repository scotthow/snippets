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


class SARIMAModel_V1:
    """
    Enhanced implementation of SARIMA model with production and test modes
    Follows the same interface as HoltWintersModel_V2
    """
    def __init__(self, seasonal_periods=16, max_tries=3):
        self.model = None
        self.fitted_model = None
        self.seasonal_periods = seasonal_periods
        self.fitted = False
        self.best_params = None
        self.train_data = None
        self.max_tries = max_tries
        
    def _find_best_params(self, data):
        """
        Find the best SARIMA parameters using grid search with convergence handling
        """
        p = d = q = range(0, 2)
        P = D = Q = range(0, 2)
        pdq = list(itertools.product(p, d, q))
        seasonal_pdq = [(x[0], x[1], x[2], self.seasonal_periods) 
                       for x in list(itertools.product(P, D, Q))]
        
        best_aic = float('inf')
        best_params = None
        best_seasonal_params = None
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            
            for param in pdq:
                for seasonal_param in seasonal_pdq:
                    for attempt in range(self.max_tries):
                        try:
                            model = SARIMAX(
                                data,
                                order=param,
                                seasonal_order=seasonal_param,
                                enforce_stationarity=False,
                                enforce_invertibility=False
                            )
                            results = model.fit(disp=False, maxiter=200)
                            
                            if results.aic < best_aic:
                                best_aic = results.aic
                                best_params = param
                                best_seasonal_params = seasonal_param
                                break
                                
                        except Exception as e:
                            logging.warning(f"SARIMAX fitting failed for parameters {param}, {seasonal_param}: {str(e)}")
                            continue
        
        # If no parameters worked, use default values
        if best_params is None:
            best_params = (1, 1, 1)
            best_seasonal_params = (1, 1, 1, self.seasonal_periods)
            logging.warning("Using default SARIMA parameters due to fitting failures")
            
        return best_params, best_seasonal_params
        
    def fit(self, data):
        """
        Fit SARIMA model with robust error handling
        """
        self.train_data = data
        
        try:
            # Find best parameters
            order, seasonal_order = self._find_best_params(data)
            self.best_params = {'order': order, 'seasonal_order': seasonal_order}
            
            # Fit model with best parameters
            self.model = SARIMAX(
                data,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            # Try different optimizers
            optimizers = ['lbfgs', 'powell', 'nm']
            fitted = False
            
            for optimizer in optimizers:
                try:
                    self.fitted_model = self.model.fit(disp=False, method=optimizer, maxiter=200)
                    fitted = True
                    break
                except Exception as e:
                    logging.warning(f"Fitting failed with optimizer {optimizer}: {str(e)}")
                    continue
            
            if not fitted:
                raise ValueError("Failed to fit model with any optimizer")
                
            self.fitted = True
            return self
            
        except Exception as e:
            logging.error(f"SARIMA model fitting failed: {str(e)}")
            raise
        
    def predict(self, steps):
        """
        Generate predictions with error handling
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        try:
            predictions = self.fitted_model.forecast(steps)
            return predictions.values  # Convert to numpy array to match HoltWinters output
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            # Return last value repeated if prediction fails
            return np.array([self.train_data[-1]] * steps)
            
    def calculate_forecast_horizon(self, cycle_id):
        """
        Calculate the forecast horizon based on the max cycle_id
        """
        year, period = map(int, cycle_id.split('-'))
        return 16 - period  # Calculate remaining periods until YYYY-16


class SARIMAModel_V2:
    """
    Enhanced SARIMA model that specifically handles end-of-cycle spikes and seasonal patterns
    """
    def __init__(self, seasonal_periods=16, max_tries=3):
        self.model = None
        self.fitted_model = None
        self.seasonal_periods = seasonal_periods
        self.fitted = False
        self.best_params = None
        self.train_data = None
        self.max_tries = max_tries
        self.cycle_end_pattern = None
        self.trend_model = None
        
    def _detect_cycle_end_pattern(self, data):
        """
        Detect and quantify the end-of-cycle spike pattern
        """
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(data)
        df.columns = ['value']
        
        # Add cycle position indicator (1-16)
        df['cycle_pos'] = np.tile(range(1, self.seasonal_periods + 1), 
                                len(df) // self.seasonal_periods + 1)[:len(df)]
        
        # Calculate average values for each position in cycle
        cycle_patterns = df.groupby('cycle_pos')['value'].agg(['mean', 'std']).reset_index()
        
        # Detect if there's a significant spike at the end of cycles
        end_positions = cycle_patterns.iloc[-3:]  # Last 3 positions
        other_positions = cycle_patterns.iloc[:-3]
        
        end_mean = end_positions['mean'].mean()
        other_mean = other_positions['mean'].mean()
        
        # Store the pattern information
        self.cycle_end_pattern = {
            'has_spike': end_mean > other_mean * 1.2,  # 20% threshold
            'spike_ratio': end_mean / other_mean if other_mean != 0 else 1,
            'position_means': cycle_patterns['mean'].values,
            'position_stds': cycle_patterns['std'].values
        }
        
    def _detrend_data(self, data):
        """
        Remove trend and seasonal components for better parameter estimation.
        Handles edge cases and missing values robustly.
        """
        # Convert data to pandas Series with index for proper decomposition
        if not isinstance(data, pd.Series):
            data = pd.Series(data)
        
        # Ensure we have enough periods for decomposition
        if len(data) < 2 * self.seasonal_periods:
            # Pad the data if necessary
            pad_length = 2 * self.seasonal_periods - len(data)
            padded_data = np.pad(data, (0, pad_length), mode='edge')
            data = pd.Series(padded_data)
        
        try:
            # Perform decomposition with robust parameters
            decomposition = seasonal_decompose(
                data,
                period=self.seasonal_periods,
                extrapolate_trend='freq',  # Use frequency-based extrapolation
                filt=None  # Disable filtering to prevent NaN generation
            )
            
            # Handle any remaining NaN values in components
            trend = pd.Series(decomposition.trend).fillna(method='ffill').fillna(method='bfill')
            seasonal = pd.Series(decomposition.seasonal).fillna(method='ffill').fillna(method='bfill')
            resid = pd.Series(decomposition.resid).fillna(method='ffill').fillna(method='bfill')
            
            # Store the cleaned components
            self.trend_model = {
                'trend': trend,
                'seasonal': seasonal,
                'resid': resid
            }
            
            # Return the cleaned residuals, trimmed to original length if padded
            return resid[:len(data)].values
            
        except Exception as e:
            logging.warning(f"Decomposition failed: {str(e)}. Falling back to simple detrending.")
            # Fallback: simple detrending using linear regression
            from sklearn.linear_model import LinearRegression
            X = np.arange(len(data)).reshape(-1, 1)
            model = LinearRegression()
            model.fit(X, data)
            trend = model.predict(X)
            resid = data - trend
            
            self.trend_model = {
                'trend': trend,
                'seasonal': np.zeros_like(data),  # No seasonal component in fallback
                'resid': resid
            }
            
            return resid
        
    def _find_best_params(self, data):
        """
        Find optimal SARIMA parameters with enhanced grid search
        """
        detrended_data = self._detrend_data(data)
        
        # Expanded parameter grid for end-cycle pattern
        if self.cycle_end_pattern['has_spike']:
            # Use higher order seasonal components if we detect end-cycle spikes
            p = d = q = range(0, 3)
            P = D = range(0, 2)
            Q = range(1, 3)  # Increased seasonal MA order
        else:
            # Standard parameter grid for normal patterns
            p = d = q = range(0, 2)
            P = D = Q = range(0, 2)
            
        pdq = list(itertools.product(p, d, q))
        seasonal_pdq = [(x[0], x[1], x[2], self.seasonal_periods) 
                       for x in list(itertools.product(P, D, Q))]
        
        best_aic = float('inf')
        best_params = None
        best_seasonal_params = None
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            
            for param in pdq:
                for seasonal_param in seasonal_pdq:
                    try:
                        model = SARIMAX(
                            detrended_data,
                            order=param,
                            seasonal_order=seasonal_param,
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        )
                        results = model.fit(disp=False, maxiter=200)
                        
                        if results.aic < best_aic:
                            best_aic = results.aic
                            best_params = param
                            best_seasonal_params = seasonal_param
                            
                    except Exception as e:
                        continue
        
        return best_params, best_seasonal_params
        
    def fit(self, data):
        """
        Fit the enhanced SARIMA model with pattern detection
        """
        self.train_data = data
        
        # Detect cycle-end patterns
        self._detect_cycle_end_pattern(data)
        
        # Find best parameters
        order, seasonal_order = self._find_best_params(data)
        self.best_params = {'order': order, 'seasonal_order': seasonal_order}
        
        # Fit final model
        try:
            self.model = SARIMAX(
                data,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            # Try different optimizers for robust fitting
            optimizers = ['lbfgs', 'powell', 'nm']
            fitted = False
            
            for optimizer in optimizers:
                try:
                    self.fitted_model = self.model.fit(disp=False, 
                                                     method=optimizer,
                                                     maxiter=200)
                    fitted = True
                    break
                except:
                    continue
            
            if not fitted:
                raise ValueError("Failed to fit model with any optimizer")
                
            self.fitted = True
            return self
            
        except Exception as e:
            logging.error(f"Enhanced SARIMA model fitting failed: {str(e)}")
            raise
            
    def predict(self, steps):
        """
        Generate predictions with cycle-end pattern adjustment
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        try:
            # Get base predictions
            predictions = self.fitted_model.forecast(steps)
            
            # Adjust predictions if we detected end-cycle patterns
            if self.cycle_end_pattern['has_spike']:
                # Calculate current position in cycle
                current_pos = len(self.train_data) % self.seasonal_periods
                
                # Adjust each prediction based on its cycle position
                for i in range(len(predictions)):
                    pos = (current_pos + i + 1) % self.seasonal_periods
                    if pos >= self.seasonal_periods - 3:  # Last 3 positions
                        # Apply spike adjustment based on historical pattern
                        predictions[i] *= self.cycle_end_pattern['spike_ratio']
            
            # return predictions.values
            return np.array(predictions) # ensure numpy array is returned (optional)
            
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            # Fallback prediction using pattern matching
            return self._fallback_predict(steps)
            
    def _fallback_predict(self, steps):
        """
        Fallback prediction method using pattern matching
        """
        last_value = self.train_data[-1]
        predictions = np.zeros(steps)
        
        for i in range(steps):
            cycle_pos = (len(self.train_data) + i) % self.seasonal_periods
            position_mean = self.cycle_end_pattern['position_means'][cycle_pos]
            predictions[i] = position_mean
            
        # Scale predictions to match the last known value
        scale_factor = last_value / predictions[0]
        predictions *= scale_factor
        
        return predictions
        
    def calculate_forecast_horizon(self, cycle_id):
        """
        Calculate the forecast horizon based on the max cycle_id
        """
        year, period = map(int, cycle_id.split('-'))
        # print(f"Forecast horizon for {cycle_id}: {16 - period}") # for QA
        return 16 - period  # Calculate remaining periods until YYYY-16



# Helper function to create and evaluate model
def evaluate_enhanced_sarima(df, test_size=7):
    """
    Create and evaluate the enhanced SARIMA model
    """
    model = SARIMAModel_V2(seasonal_periods=16)
    
    # Split data
    train_data = df['measure_rate'].iloc[:-test_size].values
    test_data = df['measure_rate'].iloc[-test_size:].values
    
    # Fit model
    model.fit(train_data)
    
    # Generate predictions
    predictions = model.predict(test_size)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(test_data, predictions))
    mae = mean_absolute_error(test_data, predictions)
    
    return {
        'model': model,
        'predictions': predictions,
        'rmse': rmse,
        'mae': mae,
        'test_data': test_data
    }

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
    

############################################################################################################
# Holt-Winters Model

class HoltWintersModel_V2:
    """
    Implementation of Holt-Winters Exponential Smoothing with damping
    Includes both test and production running modes
    """
    def __init__(self, seasonal_periods=16):
        self.model = None
        self.seasonal_periods = seasonal_periods
        self.fitted = False
        self.train_data = None
        self.metrics = {}
        
    def fit(self, data):
        """
        Fit the Holt-Winters model to the training data
        """
        self.train_data = data
        self.model = ExponentialSmoothing(
            data,
            seasonal_periods=self.seasonal_periods,
            trend='add',
            seasonal='add',
            damped_trend=True
        )
        self.fitted_model = self.model.fit(optimized=True)
        self.fitted = True
        return self
    
    def predict(self, steps):
        """
        Generate predictions for the specified number of steps ahead
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.fitted_model.forecast(steps)
        
    def calculate_forecast_horizon(self, cycle_id):
        """
        Calculate the forecast horizon based on the max cycle_id
        """
        year, period = map(int, cycle_id.split('-'))
        return 16 - period  # Calculate remaining periods until YYYY-16


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
            Type of model used (e.g., 'sarima', 'hw-damped')
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
            
            # Plot training data
            ax.plot(range(len(train_data)), train_data,
                    label='Training Data', 
                    color=colors['train'],
                    linewidth=2)
            
            # Plot test data and predictions
            test_idx = range(len(train_data), len(train_data) + len(test_data))
            ax.plot(test_idx, test_data, 
                    label='Test Data',
                    color=colors['test'],
                    linewidth=2)
            ax.plot(test_idx, predictions,
                    label='Predictions',
                    color=colors['pred'],
                    linestyle='--',
                    linewidth=2)
            
        else:  # prod mode
            train_data = df['measure_rate'].values
            
            # Plot historical data
            ax.plot(range(len(train_data)), train_data,
                    label='Historical Data',
                    color=colors['train'],
                    linewidth=2)
            
            # Plot predictions
            pred_idx = range(len(train_data), len(train_data) + len(predictions))
            ax.plot(pred_idx, predictions,
                    label='Forecast',
                    color=colors['pred'],
                    linestyle='--',
                    linewidth=2)
        
        # Format the title with model type
        model_name = model_type.upper() if model_type else ""
        if 'measure_id_key' in df.columns and 'book_name' in df.columns:
            title = f"{title} - {df['measure_id_key'].iloc[0]} ({df['book_name'].iloc[0]})\n{model_name} Model"
        
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
        model_suffix = f"_{model_type}" if model_type else ""
        plt.savefig(f'output/forecast_results_{identifier}{model_suffix}_{run_type}.png',
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
# Custom Mirror Model

class CustomMirrorModel:
    """
    Custom model that forecasts future measure_rate based on prior year's month-to-month changes,
    adjusted by the average difference between the current year's known months and the prior year's same months.
    """
    def __init__(self):
        self.fitted = False
        self.last_known_measure_rate = None
        self.alpha = None
        self.prior_year_deltas = {}
        self.cycle_ids = None
        self.measurement_years = None

    def fit(self, data):
        """
        Fit the model to the training data.

        data: pandas DataFrame with columns 'measure_rate', 'cycle_id', 'measurement_year'
        """
        self.train_data = data.reset_index(drop=True)
        self.fitted = True

        self.measure_rate = self.train_data['measure_rate'].values
        self.cycle_ids = self.train_data['cycle_id'].values
        self.measurement_years = self.train_data['measurement_year'].values

        # Extract last known measurement year and month
        last_cycle_id = self.cycle_ids[-1]
        last_year, last_period = map(int, last_cycle_id.split('-'))
        self.last_known_measure_rate = self.measure_rate[-1]

        # Calculate alpha
        # Need to compute the differences between current year's known months and prior year's same months

        # Extract current year and prior year data
        current_year = last_year
        prior_year = last_year - 1

        current_year_mask = self.measurement_years == current_year
        prior_year_mask = self.measurement_years == prior_year

        # For overlapping months
        current_months = [int(cycle_id.split('-')[1]) for cycle_id in self.cycle_ids[current_year_mask]]
        prior_months = [int(cycle_id.split('-')[1]) for cycle_id in self.cycle_ids[prior_year_mask]]

        overlapping_months = set(current_months).intersection(prior_months)

        diffs = []
        for month in overlapping_months:
            # Find index of current year data for this month
            idx_current = np.where((self.measurement_years == current_year) &
                                   (np.array([int(c.split('-')[1]) for c in self.cycle_ids]) == month))[0][0]
            idx_prior = np.where((self.measurement_years == prior_year) &
                                 (np.array([int(c.split('-')[1]) for c in self.cycle_ids]) == month))[0][0]

            rate_current = self.measure_rate[idx_current]
            rate_prior = self.measure_rate[idx_prior]

            diffs.append(rate_current - rate_prior)

        if diffs:
            self.alpha = np.mean(diffs)
        else:
            self.alpha = 0

        # Calculate month-to-month changes from prior year for forecast months
        prior_year_indices = np.where(prior_year_mask)[0]
        prior_year_cycle_ids = self.cycle_ids[prior_year_indices]
        prior_year_measure_rate = self.measure_rate[prior_year_indices]

        prior_year_months = [int(c.split('-')[1]) for c in prior_year_cycle_ids]

        # Sort prior_year_data and months by months
        sorted_indices = np.argsort(prior_year_months)
        prior_year_months_sorted = np.array(prior_year_months)[sorted_indices]
        prior_year_measure_rate_sorted = prior_year_measure_rate[sorted_indices]

        # Compute month-to-month deltas
        prior_year_deltas = prior_year_measure_rate_sorted[1:] - prior_year_measure_rate_sorted[:-1]

        prior_year_months_sorted = prior_year_months_sorted[1:]  # Adjust months accordingly

        # Store deltas with months
        self.prior_year_deltas = dict(zip(prior_year_months_sorted, prior_year_deltas))

    def calculate_forecast_horizon(self, cycle_id):
        """
        Calculate the forecast horizon based on the max cycle_id
        """
        year, period = map(int, cycle_id.split('-'))
        return 16 - period  # Calculate remaining periods until YYYY-16

    def predict(self, steps):
        """
        Generate predictions for the specified number of steps ahead
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Starting from the last known measure_rate
        predictions = []

        last_measure_rate = self.last_known_measure_rate

        # Get the last cycle_id, and compute forecast months
        last_cycle_id = self.cycle_ids[-1]
        last_year, last_period = map(int, last_cycle_id.split('-'))

        # Forecast periods: steps ahead
        forecast_periods = []
        for step in range(1, steps + 1):
            next_period = last_period + step
            next_year = last_year
            if next_period > 16:
                next_period -= 16
                next_year += 1
            forecast_periods.append((next_year, next_period))

        for year, period in forecast_periods:
            # The corresponding prior year's period is period
            prior_year = year - 1
            prior_period = period

            # Get delta from prior year for this period
            delta = self.prior_year_deltas.get(prior_period, 0)  # If no data, assume zero delta

            # Compute the forecasted value
            forecast_value = last_measure_rate + delta + self.alpha

            predictions.append(forecast_value)

            # Update last_measure_rate for next iteration
            last_measure_rate = forecast_value

        return predictions
    

# ############################################################################################################
# # Utility function to run the entire prediction process

# def make_predictions(df, run_type="test", test_size=7, model_class=HoltWintersModel_V2):
#     """
#     Utility function to run the entire prediction process
#     Now accepts model_class parameter to specify which model to use
#     """
#     if 'predictions' not in df.columns:
#         df['predictions'] = None
    
#     model = model_class(seasonal_periods=16)
#     evaluator = ModelEvaluator()
    
#     if run_type == "test":
#         train_df = df.iloc[:-test_size]
#         test_data = df['measure_rate'].iloc[-test_size:].values
        
#         model.fit(train_df['measure_rate'].values)
#         metrics = evaluator.evaluate(model, train_df, test_data, run_type="test")
        
#         results_df = evaluator.create_results_df(
#             df,
#             metrics['predictions'],
#             run_type="test",
#             test_data=test_data
#         )
        
#         evaluator.plot_results(
#             df,
#             metrics['predictions'],
#             run_type="test",
#             test_data=test_data
#         )
        
#     else:  # prod mode
#         model.fit(df['measure_rate'].values)
#         metrics = evaluator.evaluate(model, df, run_type="prod")
        
#         results_df = evaluator.create_results_df(
#             df,
#             metrics['predictions'],
#             run_type="prod"
#         )
        
#         evaluator.plot_results(
#             df,
#             metrics['predictions'],
#             run_type="prod"
#         )
    
#     return results_df, model, evaluator

############################################################################################################
# Utility function to run the entire prediction process

def make_predictions(df, run_type="test", test_size=7, model_id='hw-damped'):
    """
    Utility function to run the entire prediction process
    Now selects model_class based on model_id parameter
    """
    if 'predictions' not in df.columns:
        df['predictions'] = None

    # Map model_id to model_class
    model_classes = {
        'hw-damped': HoltWintersModel_V2,
        'sarima': SARIMAModel_V2,
        'custom-mirror': CustomMirrorModel
    }

    if model_id not in model_classes:
        raise ValueError(f"Model type {model_id} not recognized")

    model_class = model_classes[model_id]

    # Instantiate the model
    if model_id == 'hw-damped':
        model = model_class(seasonal_periods=16)
    else:
        model = model_class()

    evaluator = ModelEvaluator()

    if run_type == "test":
        train_df = df.iloc[:-test_size]
        test_data = df['measure_rate'].iloc[-test_size:].values

        # For models that require the entire DataFrame
        if model_id == 'custom-mirror':
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
        if model_id == 'custom-mirror':
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