# Beginning of the new measurement year where we don't yet have reported rates
elif HYBRID_PERIOD == 0:
    logging.info(f'{population} - {measure} in HYBRID PERIOD {HYBRID_PERIOD}')
    
    # Get all the possible cycle_ids up to the max one in the dataframe
    max_cycle_id = df['cycle_id'].max()
    max_year = int(max_cycle_id.split('-')[0])
    max_period = int(max_cycle_id.split('-')[1])
    
    # Generate all expected cycle_ids that should exist up to the max one
    expected_cycle_ids = []
    for year in range(int(MEASUREMENT_YEAR) - 1, max_year + 1):
        for period in range(1, 17):
            cycle_id = f"{year}-{period:02d}"
            # Only include cycle_ids up to the max one
            if (year < max_year) or (year == max_year and period <= max_period):
                expected_cycle_ids.append(cycle_id)
    
    # Sort expected_cycle_ids to ensure we process them in chronological order
    # This is critical to ensure we can build on synthetic rows we create
    expected_cycle_ids.sort()
    
    # Check for any missing cycle_ids and create synthetic rows
    synthetic_rows = []
    for cycle_id in expected_cycle_ids:
        if cycle_id not in df['cycle_id'].values:
            # Get the previous cycle_id to copy from
            year = int(cycle_id.split('-')[0])
            period = int(cycle_id.split('-')[1])
            
            if period == 1:
                # For first period in a cycle (XX-01), find the previous year's last period
                prev_cycle_id = f"{year-1}-16"
                
                # Check if the previous cycle exists in original df or newly created synthetic rows
                existing_df = pd.concat([df, pd.DataFrame(synthetic_rows)], ignore_index=True) if synthetic_rows else df
                
                if prev_cycle_id in existing_df['cycle_id'].values:
                    row_to_copy = existing_df[existing_df['cycle_id'] == prev_cycle_id].copy().iloc[0]
                    row_to_copy['cycle_id'] = cycle_id
                    row_to_copy['model_id'] = model_id  # IMPORTANT: ensure the proper model id
                    row_to_copy['synthetic'] = 1
                    row_to_copy['measure_rate'] = 0  # Set measure_rate to zero for first in cycle
                else:
                    # Create a template row from any existing row
                    template_row = df.iloc[0].copy()
                    row_to_copy = template_row.copy()
                    row_to_copy['cycle_id'] = cycle_id
                    row_to_copy['model_id'] = model_id
                    row_to_copy['synthetic'] = 1
                    row_to_copy['measure_rate'] = 0  # Set measure_rate to zero for first in cycle
            else:
                # For other periods, find the previous period
                prev_cycle_id = f"{year}-{period-1:02d}"
                
                # Check in both original df and newly created synthetic rows
                existing_df = pd.concat([df, pd.DataFrame(synthetic_rows)], ignore_index=True) if synthetic_rows else df
                
                if prev_cycle_id in existing_df['cycle_id'].values:
                    row_to_copy = existing_df[existing_df['cycle_id'] == prev_cycle_id].copy().iloc[0]
                    row_to_copy['cycle_id'] = cycle_id
                    row_to_copy['model_id'] = model_id
                    row_to_copy['synthetic'] = 1
                else:
                    # If previous period is also missing, go back until we find an existing cycle_id
                    found_previous = False
                    for p in range(period-2, 0, -1):
                        earlier_cycle_id = f"{year}-{p:02d}"
                        if earlier_cycle_id in existing_df['cycle_id'].values:
                            row_to_copy = existing_df[existing_df['cycle_id'] == earlier_cycle_id].copy().iloc[0]
                            row_to_copy['cycle_id'] = cycle_id
                            row_to_copy['model_id'] = model_id
                            row_to_copy['synthetic'] = 1
                            found_previous = True
                            break
                    
                    # If no previous period found in the same year, check the previous year
                    if not found_previous:
                        prev_year_cycle_id = f"{year-1}-16"
                        if prev_year_cycle_id in existing_df['cycle_id'].values:
                            row_to_copy = existing_df[existing_df['cycle_id'] == prev_year_cycle_id].copy().iloc[0]
                            row_to_copy['cycle_id'] = cycle_id
                            row_to_copy['model_id'] = model_id
                            row_to_copy['synthetic'] = 1
                        else:
                            # If all else fails, use the first row as a template
                            template_row = df.iloc[0].copy()
                            row_to_copy = template_row.copy()
                            row_to_copy['cycle_id'] = cycle_id
                            row_to_copy['model_id'] = model_id
                            row_to_copy['synthetic'] = 1
            
            # Add the synthetic row to our collection
            synthetic_rows.append(row_to_copy)
    
    # Append all synthetic rows to the dataframe at once
    if synthetic_rows:
        synthetic_df = pd.DataFrame(synthetic_rows)
        df = pd.concat([df, synthetic_df], ignore_index=True)