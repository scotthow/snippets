# Beginning of the new measurement year where we don't yet have reported rates
elif HYBRID_PERIOD == 0:
    logging.info(f'{population} - {measure} in HYBRID PERIOD {HYBRID_PERIOD}')
    
    # Find all unique measurement years in the dataframe
    all_years = set()
    for cycle in df['cycle_id'].values:
        year = cycle.split('-')[0]
        all_years.add(int(year))
    
    # Get the max cycle_id
    max_cycle_id = df['cycle_id'].max()
    max_year, max_period = int(max_cycle_id.split('-')[0]), int(max_cycle_id.split('-')[1])
    
    # Generate all expected cycle_ids that should exist
    expected_cycle_ids = []
    for year in sorted(all_years):
        # For each year, add all periods (01-16) unless it's the max year
        periods_to_check = 16
        if year == max_year:
            periods_to_check = max_period
            
        for period in range(1, periods_to_check + 1):
            cycle_id = f"{year}-{period:02d}"
            expected_cycle_ids.append(cycle_id)
    
    # Create a set of existing cycle_ids for faster lookup
    existing_cycle_ids = set(df['cycle_id'].values)
    
    # Find missing cycle_ids
    missing_cycle_ids = [cid for cid in expected_cycle_ids if cid not in existing_cycle_ids]
    
    # Sort missing cycle_ids to ensure we process them in chronological order
    missing_cycle_ids.sort()
    
    # Create synthetic rows for all missing cycle_ids
    for cycle_id in missing_cycle_ids:
        year, period = int(cycle_id.split('-')[0]), int(cycle_id.split('-')[1])
        
        if period == 1:
            # For first period in a cycle (XX-01)
            prev_cycle_id = f"{year-1}-16"
            
            if prev_cycle_id in existing_cycle_ids:
                row_to_copy = df[df['cycle_id'] == prev_cycle_id].copy()
            else:
                # If previous cycle doesn't exist, use any existing row as a template
                row_to_copy = df.iloc[0:1].copy()
            
            # Update the row attributes
            row_to_copy['cycle_id'] = cycle_id
            row_to_copy['model_id'] = model_id
            row_to_copy['synthetic'] = 1
            row_to_copy['measure_rate'] = 0  # Set measure_rate to zero for first in cycle
        else:
            # For other periods, find the previous period
            prev_cycle_id = f"{year}-{period-1:02d}"
            
            if prev_cycle_id in existing_cycle_ids:
                row_to_copy = df[df['cycle_id'] == prev_cycle_id].copy()
            else:
                # If the immediate previous period is also missing,
                # find the latest available period before this one
                available_periods = []
                for p in range(1, period):
                    check_cycle = f"{year}-{p:02d}"
                    if check_cycle in existing_cycle_ids:
                        available_periods.append(check_cycle)
                
                if available_periods:
                    # Use the latest available period in the same year
                    latest_available = sorted(available_periods)[-1]
                    row_to_copy = df[df['cycle_id'] == latest_available].copy()
                else:
                    # If no periods available in this year, try the last period of the previous year
                    prev_year_last = f"{year-1}-16"
                    if prev_year_last in existing_cycle_ids:
                        row_to_copy = df[df['cycle_id'] == prev_year_last].copy()
                    else:
                        # If all else fails, use any row as a template
                        row_to_copy = df.iloc[0:1].copy()
            
            # Update the row attributes
            row_to_copy['cycle_id'] = cycle_id
            row_to_copy['model_id'] = model_id
            row_to_copy['synthetic'] = 1
        
        # Append the new row to the dataframe
        df = pd.concat([df, row_to_copy], ignore_index=True)
        
        # Update our set of existing cycle_ids to include this new synthetic one
        existing_cycle_ids.add(cycle_id)