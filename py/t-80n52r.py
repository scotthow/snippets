import pandas as pd

def process_measure_rates(df):
    # Create a proper sortable version of cycle_id
    df['sort_key'] = df['cycle_id'].apply(lambda x: tuple(map(int, x.split('-'))))
    
    # Get max cycle_id for each book_name and measure_id_key combination
    idx = df.groupby(['book_name', 'measure_id_key'])['sort_key'].idxmax()
    
    # Create result DataFrame with max cycle_id rows
    result_df = df.loc[idx].copy()
    
    # Add curr_cycle_id and curr_rate columns
    result_df['curr_cycle_id'] = result_df['cycle_id']
    result_df['curr_rate'] = result_df['measure_rate']
    
    # Function to get measure_rate for a specific cycle_id
    def get_rate_for_cycle(row, target_cycle_id):
        mask = (df['book_name'] == row['book_name']) & \
               (df['measure_id_key'] == row['measure_id_key']) & \
               (df['cycle_id'] == target_cycle_id)
        rates = df.loc[mask, 'measure_rate']
        return rates.iloc[0] if not rates.empty else None
    
    # Extract year and period from current cycle_id
    result_df['curr_year'] = result_df['curr_cycle_id'].str.split('-').str[0].astype(int)
    result_df['curr_period'] = result_df['curr_cycle_id'].str.split('-').str[1].astype(int)
    
    # Add prior year rate columns
    result_df['curr_rate_prior_year2'] = result_df.apply(
        lambda row: get_rate_for_cycle(
            row, f"{row['curr_year']-2}-{row['curr_period']:02d}"
        ),
        axis=1
    )
    
    result_df['curr_rate_prior_year'] = result_df.apply(
        lambda row: get_rate_for_cycle(
            row, f"{row['curr_year']-1}-{row['curr_period']:02d}"
        ),
        axis=1
    )
    
    # Add last admin rate columns
    result_df['prior_year_last_admin2'] = result_df.apply(
        lambda row: get_rate_for_cycle(
            row, f"{row['curr_year']-2}-15"
        ),
        axis=1
    )
    
    result_df['prior_year_last_admin'] = result_df.apply(
        lambda row: get_rate_for_cycle(
            row, f"{row['curr_year']-1}-15"
        ),
        axis=1
    )
    
    # Add reported rate columns
    result_df['prior_year_reported_rate2'] = result_df.apply(
        lambda row: get_rate_for_cycle(
            row, f"{row['curr_year']-2}-16"
        ),
        axis=1
    )
    
    result_df['prior_year_reported_rate'] = result_df.apply(
        lambda row: get_rate_for_cycle(
            row, f"{row['curr_year']-1}-16"
        ),
        axis=1
    )
    
    # Drop temporary columns
    result_df.drop(['sort_key', 'curr_year', 'curr_period'], axis=1, inplace=True)
    
    return result_df