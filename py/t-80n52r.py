import pandas as pd
import numpy as np

def calculate_baseline_rate(row, per_change_threshold, diff_threshold, diff_window, curr_rate_weight, avg_diff_weight):
    """
    Calculates a baseline rate based on historical data and current performance.
    
    Parameters:
        row (pd.Series): A single row from the DataFrame.
        per_change_threshold (float): Threshold for absolute % change.
        diff_threshold (float): Threshold for stability of year-over-year deltas.
        diff_window (int): Minimum number of years to consider for avg diff calculation.
        curr_rate_weight (float): Weight for current rate in baseline calculation.
        avg_diff_weight (float): Weight for average difference in baseline calculation.
        
    Returns:
        float: A baseline rate.
    """
    
    # Normalize weights just in case (optional step)
    total_weight = curr_rate_weight + avg_diff_weight
    if total_weight == 0:
        # Default to curr_rate if weights are zero
        curr_rate_weight = 1.0
        avg_diff_weight = 0.0
    else:
        curr_rate_weight = curr_rate_weight / total_weight
        avg_diff_weight = avg_diff_weight / total_weight

    # Extract values
    curr_rate = row.get('curr_rate', None)
    pyr_rate = row.get('prior_year_reported_rate', None)
    pyr_rate2 = row.get('prior_year_reported_rate2', None)
    curr_rate_pyr = row.get('curr_rate_prior_year', None)
    curr_rate_pyr2 = row.get('curr_rate_prior_year2', None)

    yearly_deltas = []
    if (pyr_rate is not None) and (curr_rate_pyr is not None):
        yearly_deltas.append(pyr_rate - curr_rate_pyr)
    if (pyr_rate2 is not None) and (curr_rate_pyr2 is not None):
        yearly_deltas.append(pyr_rate2 - curr_rate_pyr2)

    # Compute avg_diff if we have enough data
    avg_diff = None
    if len(yearly_deltas) >= diff_window:
        avg_diff_raw = np.mean(yearly_deltas)
        avg_abs_diff_deltas = np.mean(np.abs(np.diff(yearly_deltas))) if len(yearly_deltas) > 1 else 0
        
        # If differences are not stable, default to the first delta
        if avg_abs_diff_deltas > diff_threshold:
            avg_diff = yearly_deltas[0]
        else:
            avg_diff = avg_diff_raw

    # Determine baseline
    if avg_diff is not None:
        # Weighted combination of current rate and avg_diff
        if curr_rate is None:
            # If no current rate, fallback to prior year reported rate
            curr_rate = pyr_rate if pyr_rate is not None else 0.0
        baseline_rate = curr_rate_weight * curr_rate + avg_diff_weight * (curr_rate + avg_diff)
    else:
        # If we don't have enough data for avg_diff, fallback:
        # Prefer current rate if available, else prior year's rate, else 0.0
        if curr_rate is not None:
            baseline_rate = curr_rate
        elif pyr_rate is not None:
            baseline_rate = pyr_rate
        else:
            baseline_rate = 0.0

    return baseline_rate

def improved_forecast_adjustment(row, per_change_threshold, diff_threshold, diff_window, curr_rate_weight, avg_diff_weight):
    """
    Adjusts the forecasted rate based on historical data, current performance, and a baseline rate.

    Parameters:
        row (pd.Series): A single row of the DataFrame.
        per_change_threshold (float): Threshold for absolute % change.
        diff_threshold (float): Threshold for yoy delta stability.
        diff_window (int): Number of years for avg_diff calculation.
        curr_rate_weight (float): Weight for curr_rate in baseline.
        avg_diff_weight (float): Weight for avg_diff in baseline.

    Returns:
        float: An adjusted forecasted rate.
    """

    model_id = row.get('model_id', None)
    curr_rate = row.get('curr_rate', None)
    forecasted_rate = row.get('forecasted_rate', None)
    pyr_rate = row.get('prior_year_reported_rate', None)

    # Pass-through model logic
    if model_id == 'pass_through':
        if (pyr_rate is not None) and (curr_rate is not None):
            return max(curr_rate, pyr_rate)
        elif curr_rate is not None:
            return curr_rate
        else:
            return forecasted_rate if forecasted_rate is not None else 0.0

    # Calculate baseline rate
    baseline_rate = calculate_baseline_rate(
        row, per_change_threshold, diff_threshold, diff_window, curr_rate_weight, avg_diff_weight
    )

    # Compute percentage change from prior year's reported rate
    if (pyr_rate is not None) and (pyr_rate != 0) and (forecasted_rate is not None):
        per_change = (forecasted_rate - pyr_rate) / pyr_rate
    else:
        per_change = None

    # For certain models, adjust forecast if it deviates too far
    if model_id in ['hw_damped', 'hw_nd', 'custom_mirror']:
        if per_change is not None and abs(per_change) > per_change_threshold:
            # Instead of fully reverting to baseline_rate, consider blending them:
            # This makes the adjustment smoother
            alpha = 0.5  # Blend factor
            adjusted_rate = alpha * forecasted_rate + (1 - alpha) * baseline_rate
            return adjusted_rate
        else:
            return forecasted_rate if forecasted_rate is not None else baseline_rate
    else:
        # For other models not specified, ensure the adjusted forecast
        # is at least the current rate if that makes sense:
        if forecasted_rate is not None and curr_rate is not None:
            return max(curr_rate, forecasted_rate)
        elif forecasted_rate is not None:
            return forecasted_rate
        else:
            return baseline_rate

# Example configuration:
per_change_threshold = 0.02   # Allow ±2% deviation
diff_threshold = 0.04         # Threshold for yoy delta stability
diff_window = 2               # Require at least 2 years of data
curr_rate_weight = 0.7        # 70% weight to current rate
avg_diff_weight = 0.3         # 30% weight to historical differences

# Apply the improved adjustment function
fc_df['forecasted_rate_adj'] = fc_df.apply(
    lambda row: improved_forecast_adjustment(
        row, per_change_threshold, diff_threshold, diff_window, curr_rate_weight, avg_diff_weight
    ),
    axis=1
)
