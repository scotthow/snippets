# Beginning of the new measurement year where we don't yet have reported rates
elif HYBRID_PERIOD == 0:
    logging.info(f'{population} - {measure} in HYBRID PERIOD {HYBRID_PERIOD}')
    
    # Get all the possible cycle_ids up to the max one in the dataframe
    max_cycle_id = df['cycle_id'].max()
    max_year, max_period = int(max_cycle_id.split('-')[0]), int(max_cycle_id.split('-')[1])
    
    # Generate all expected cycle_ids that should exist up to the max one
    expected_cycle_ids = []
    for year in range(int(MEASUREMENT_YEAR) - 1, max_year + 1):
        for period in range(1, 17):
            cycle_id = f"{year}-{period:02d}"
            # Only include cycle_ids up to the max one
            if (year < max_year) or (year == max_year and period <= max_period):
                expected_cycle_ids.append(cycle_id)
    
    # Sort expected_cycle_ids to ensure we process them in chronological order
    expected_cycle_ids.sort()
    
    # Create a set of existing cycle_ids for faster lookup
    existing_cycle_ids = set(df['cycle_id'].values)
    
    # Check for any missing cycle_ids and create synthetic rows
    synthetic_rows = []
    
    for cycle_id in expected_cycle_ids:
        if cycle_id not in existing_cycle_ids:
            year, period = int(cycle_id.split('-')[0]), int(cycle_id.split('-')[1])
            
            if period == 1:
                # For first period in a cycle (XX-01)
                prev_cycle_id = f"{year-1}-16"
                
                # Check if we have the previous cycle_id (either original or synthetic)
                if prev_cycle_id in existing_cycle_ids:
                    row_to_copy = df[df['cycle_id'] == prev_cycle_id].copy()
                elif any(r['cycle_id'] == prev_cycle_id for r in synthetic_rows):
                    # Get from our new synthetic rows if not in original df
                    row_to_copy = next(pd.DataFrame([r]) for r in synthetic_rows if r['cycle_id'] == prev_cycle_id)
                else:
                    # Create a template row if we don't have the previous cycle
                    row_to_copy = pd.DataFrame([df.iloc[0].copy()])
                
                # Update the row attributes
                row_to_copy['cycle_id'] = cycle_id
                row_to_copy['model_id'] = model_id
                row_to_copy['synthetic'] = 1
                row_to_copy['measure_rate'] = 0  # Set measure_rate to zero for first in cycle
            else:
                # For other periods, find the previous period
                prev_cycle_id = f"{year}-{period-1:02d}"
                
                # Check if we have the previous cycle_id (either original or synthetic)
                if prev_cycle_id in existing_cycle_ids:
                    row_to_copy = df[df['cycle_id'] == prev_cycle_id].copy()
                elif any(r['cycle_id'] == prev_cycle_id for r in synthetic_rows):
                    # Get from our new synthetic rows if not in original df
                    for r in synthetic_rows:
                        if r['cycle_id'] == prev_cycle_id:
                            row_to_copy = pd.DataFrame([r.copy()])
                            break
                else:
                    # If previous period is also missing, find the latest available period
                    available_periods = [
                        cid for cid in existing_cycle_ids.union({r['cycle_id'] for r in synthetic_rows})
                        if cid.startswith(f"{year}-") and int(cid.split('-')[1]) < period
                    ]
                    
                    if available_periods:
                        latest_period = max(available_periods, key=lambda x: int(x.split('-')[1]))
                        if latest_period in existing_cycle_ids:
                            row_to_copy = df[df['cycle_id'] == latest_period].copy()
                        else:
                            for r in synthetic_rows:
                                if r['cycle_id'] == latest_period:
                                    row_to_copy = pd.DataFrame([r.copy()])
                                    break
                    else:
                        # If no period is available for this year, get the last period of previous year
                        prev_year_latest = f"{year-1}-16"
                        if prev_year_latest in existing_cycle_ids:
                            row_to_copy = df[df['cycle_id'] == prev_year_latest].copy()
                        elif any(r['cycle_id'] == prev_year_latest for r in synthetic_rows):
                            for r in synthetic_rows:
                                if r['cycle_id'] == prev_year_latest:
                                    row_to_copy = pd.DataFrame([r.copy()])
                                    break
                        else:
                            # Create a template row if all else fails
                            row_to_copy = pd.DataFrame([df.iloc[0].copy()])
            
                # Update the row attributes
                row_to_copy['cycle_id'] = cycle_id
                row_to_copy['model_id'] = model_id
                row_to_copy['synthetic'] = 1
            
            # Add to our list of synthetic rows
            if isinstance(row_to_copy, pd.DataFrame):
                synthetic_row = row_to_copy.iloc[0].to_dict()
                synthetic_rows.append(synthetic_row)
            
            # Update our set of existing cycle_ids to include this new synthetic one
            existing_cycle_ids.add(cycle_id)
    
    # Append all synthetic rows to the dataframe
    if synthetic_rows:
        synthetic_df = pd.DataFrame(synthetic_rows)
        df = pd.concat([df, synthetic_df], ignore_index=True)