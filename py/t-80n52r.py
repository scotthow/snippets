class CustomMirrorModel:
    """
    A custom forecasting model that combines historical patterns with current trends
    to predict future measure rates while maintaining reasonable proximity to prior year values.
    Includes enhanced trend analysis for more realistic final rate predictions.
    """
    def __init__(self):
        self.train_data = None
        self.fitted = False
        self.prior_year_data = None
        self.current_year_data = None
        self.trend_coefficient = None
        self.seasonal_patterns = None
        self.current_trend = None
        self.prior_trend = None
        
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
        
    def _calculate_trends(self):
        """
        Calculate and compare trends for current and prior year data
        """
        # Calculate prior year trend using all periods
        prior_rates = self.prior_year_data['measure_rate']
        prior_periods = self.prior_year_data['period']
        self.prior_trend = np.polyfit(prior_periods, prior_rates, 1)[0]
        
        # Calculate current year trend using available periods
        current_rates = self.current_year_data['measure_rate']
        current_periods = self.current_year_data['period']
        self.current_trend = np.polyfit(current_periods, current_rates, 1)[0]
        
        # Calculate trend difference percentage
        trend_diff_pct = (self.current_trend - self.prior_trend) / abs(self.prior_trend)
        return trend_diff_pct
        
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
        
        # Calculate trends first
        trend_diff_pct = self._calculate_trends()
        
        # Calculate trend coefficient and seasonal patterns
        self._calculate_trend_coefficient()
        self._calculate_seasonal_patterns()
        
        self.fitted = True
        return self
        
    def predict(self, steps):
        """
        Generate predictions for the specified number of steps ahead
        with enhanced trend-based adjustment for final periods
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
            steps_to_end = 16 - target_period
            dampening_factor = max(0.4, steps_to_end / 16)
            
            # Apply progressively dampened trend adjustment
            adjusted_trend = self.trend_coefficient * dampening_factor
            predicted_value = prior_value * (1 + adjusted_trend)
            
            # Apply additional trend-based adjustment for final periods (13-16)
            if target_period >= 13:
                # Calculate trend-based adjustment factor
                trend_diff = self.current_trend - self.prior_trend
                trend_adjustment = np.clip(trend_diff * 2.5, -0.1, 0.1)  # Scale factor of 2.5
                
                # Apply stronger adjustment for final period (16)
                if target_period == 16:
                    trend_adjustment *= 1.5  # 50% stronger for final period
                
                # Calculate base proximity factor
                base_proximity = 1 - ((target_period - 12) / 4) * 0.6
                
                # Adjust proximity factor based on trend difference
                if trend_diff < 0:  # Current trend is lower than prior
                    proximity_factor = base_proximity * (1 + abs(trend_adjustment))
                else:  # Current trend is higher than prior
                    proximity_factor = base_proximity * (1 - trend_adjustment)
                
                # Apply adjusted proximity factor
                predicted_value = (predicted_value * proximity_factor + 
                                 prior_value * (1 - proximity_factor))
                
                # Apply final trend adjustment for period 16
                if target_period == 16 and trend_diff < 0:
                    trend_scale = min(abs(trend_diff) * 3, 0.15)  # Cap at 15% adjustment
                    predicted_value *= (1 - trend_scale)
            
            # Apply smoothing for transition from last known value
            if i == 0 and last_period < 16:
                last_known_value = self.current_year_data[
                    self.current_year_data['period'] == last_period
                ]['measure_rate'].iloc[0]
                alpha = 0.6
                predicted_value = (alpha * predicted_value + 
                                 (1 - alpha) * last_known_value)
            
            predictions.append(predicted_value)
            
        return np.array(predictions)
        
    def calculate_forecast_horizon(self, cycle_id):
        """
        Calculate the forecast horizon based on the max cycle_id
        """
        year, period = map(int, cycle_id.split('-'))
        return 16 - period