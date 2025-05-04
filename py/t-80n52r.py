# Beginning of the new measurement year where we don't yet have reported rates
elif HYBRID_PERIOD == 0:
    logging.info(f'{population} - {measure} in HYBRID PERIOD {HYBRID_PERIOD}')
    
    # Get the current year
    current_year = MEASUREMENT_YEAR
    
    # Create all possible cycle_ids for the current measurement year
    # We always check all 16 cycle_ids for the current measurement year
    all_possible_cycle_ids = [f"{current_year}-{str(i).zfill(2)}" for i in range(1, 17)]
    
    # Find missing cycle_ids
    existing_cycle_ids = df['cycle_id'].values
    missing_cycle_ids = [cycle_id for cycle_id in all_possible_cycle_ids if cycle_id not in existing_cycle_ids]
    
    # For each missing cycle_id, copy the preceding row
    for missing_id in missing_cycle_ids:
        # Extract the month number
        month_num = int(missing_id.split('-')[1])
        
        # For cycle_ids ending in "01", set a special zero value row
        if month_num == 1:
            # Create a zero value row for the first month
            if len(df) > 0:
                # Use structure of an existing row but zero out the values
                row_to_copy = df.iloc[0].copy()
                # Keep only structural columns, zero out metrics
                for col in row_to_copy.index:
                    if col not in ['cycle_id', 'model_id', 'synthetic']:
                        try:
                            float(row_to_copy[col])  # Check if it's a numeric column
                            row_to_copy[col] = 0
                        except:
                            pass  # Keep non-numeric columns as is
            else:
                # If no rows exist, create a basic row with zeros
                row_to_copy = pd.Series({'synthetic': 1, 'model_id': model_id})
            
            # Update cycle_id
            row_to_copy['cycle_id'] = missing_id
            row_to_copy['synthetic'] = 1
            row_to_copy['model_id'] = model_id
        else:
            # For all other months, copy the preceding month
            prev_cycle_id = f"{current_year}-{str(month_num-1).zfill(2)}"
            
            # Check if we have the preceding cycle_id
            if prev_cycle_id in existing_cycle_ids:
                # Copy the row with the preceding cycle_id
                row_to_copy = df[df['cycle_id'] == prev_cycle_id].copy().iloc[0]
                # Update the cycle_id
                row_to_copy['cycle_id'] = missing_id
                # Mark as synthetic
                row_to_copy['synthetic'] = 1
                # Ensure proper model_id
                row_to_copy['model_id'] = model_id
            else:
                logging.warning(f"Cannot create synthetic entry for {missing_id}: preceding cycle_id {prev_cycle_id} also missing")
                continue
        
        # Convert row_to_copy to DataFrame if it's a Series
        if isinstance(row_to_copy, pd.Series):
            row_to_copy = pd.DataFrame([row_to_copy])
        
        # Append row to original df
        df = pd.concat([df, row_to_copy], ignore_index=True)
        
        logging.info(f"Created synthetic entry for missing cycle_id {missing_id}")