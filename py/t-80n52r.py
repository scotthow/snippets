def handle_missing_AMO(self, target_measure_ids=None, verbose=True):
    """
    Handles missing Annual Measurement Only (AMO) cases:
    1. For 2023-01 through 2023-03: Creates synthetic rows based on 2023-04 data with adjusted rates
    2. For 2023-16: Creates synthetic row based on 2023-15 data with same rate
    
    Args:
        target_measure_ids (list, optional): List of measure IDs to process.
            Defaults to ['901715'] if None provided.
        verbose (bool, optional): If True, prints informative messages about processing.
            Defaults to True.
            
    Returns:
        bool: True if processing was performed, False if no processing needed
    
    Raises:
        ValueError: If required columns are missing
    """
    # Validate required columns
    required_cols = [
        'measure_id_key', 'cycle_id', 'measurement_year', 'measure_rate',
        'synthetic', 'inverse_measure_flag', 'lower_rate_is_better_flag',
        'instance_data_through_dt'
    ]
    missing_cols = [col for col in required_cols if col not in self.df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Set default target measure IDs if none provided
    if target_measure_ids is None:
        target_measure_ids = ['901715']
    
    # Convert target_measure_ids to list if single string provided
    if isinstance(target_measure_ids, str):
        target_measure_ids = [target_measure_ids]
    
    # Check if current measure_id_key needs processing
    current_measure_id = str(self.df['measure_id_key'].iloc[0])
    
    if current_measure_id not in target_measure_ids:
        if verbose:
            print(f"Measure ID {current_measure_id} is not in target list {target_measure_ids}. "
                  f"No processing performed. DataFrame remains unchanged.")
        return False
    
    if verbose:
        print(f"Processing measure ID {current_measure_id}")
        
    # Ensure DataFrame is using the correct index
    if not isinstance(self.df.index, pd.MultiIndex):
        self._create_time_components()
    
    original_row_count = len(self.df)
    new_rows = []
    
    # First, handle period 16 if missing
    source_idx_16 = (2023, 15)  # Source row for period 16
    target_idx_16 = (2023, 16)  # Target period 16
    
    if target_idx_16 not in self.df.index and source_idx_16 in self.df.index:
        source_row_16 = self.df.loc[source_idx_16].copy()
        source_row_16['cycle_id'] = "2023-16"
        source_row_16['synthetic'] = 1
        source_row_16['instance_data_through_dt'] = None
        
        # Create new index and DataFrame for period 16
        new_idx_16 = pd.MultiIndex.from_tuples([target_idx_16], names=['year', 'period'])
        new_df_16 = pd.DataFrame([source_row_16])
        new_df_16.index = new_idx_16
        new_rows.append(new_df_16)
        
        if verbose:
            print(f"Created synthetic row for cycle 2023-16 with "
                  f"measure rate {source_row_16['measure_rate']:.4f}")
    
    # Now handle periods 01-03
    source_idx = (2023, 4)  # Source row for periods 01-03
    if source_idx not in self.df.index:
        if verbose:
            print(f"Cycle 2023-04 not found for measure ID {current_measure_id}. "
                  f"Cannot create synthetic rows for periods 01-03.")
        # Continue processing as we might have added period 16
    else:
        source_row = self.df.loc[source_idx].copy()
        
        # Determine if we should add or subtract based on flags
        inverse_logic = (
            source_row['inverse_measure_flag'] == 1 or 
            source_row['lower_rate_is_better_flag'] == 1
        )
        
        base_rate = source_row['measure_rate']
        
        # Create synthetic rows for periods 1-3
        for period in range(3, 0, -1):
            # Skip if row already exists
            if (2023, period) in self.df.index:
                if verbose:
                    print(f"Cycle 2023-{period:02d} already exists. Skipping.")
                continue
                
            # Create new row
            new_row = source_row.copy()
            new_row['cycle_id'] = f"2023-{period:02d}"
            
            # Calculate new measure rate
            steps_from_04 = 4 - period  # Number of steps back from period 04
            if inverse_logic:
                adjustment = 0.01 * steps_from_04  # Add 1 percentage point per step
            else:
                adjustment = -0.01 * steps_from_04  # Subtract 1 percentage point per step
            
            new_row['measure_rate'] = base_rate + adjustment
            new_row['synthetic'] = 1
            new_row['instance_data_through_dt'] = None
            
            # Create new index and DataFrame
            new_idx = pd.MultiIndex.from_tuples([(2023, period)], names=['year', 'period'])
            new_df = pd.DataFrame([new_row])
            new_df.index = new_idx
            new_rows.append(new_df)
            
            if verbose:
                print(f"Created synthetic row for cycle {new_row['cycle_id']} with "
                      f"measure rate {new_row['measure_rate']:.4f}")
    
    # Add all new rows to DataFrame if any were created
    if new_rows:
        self.df = pd.concat([self.df] + new_rows)
        self.df.sort_index(inplace=True)
        
        if verbose:
            rows_added = len(self.df) - original_row_count
            print(f"Processing complete. Added {rows_added} synthetic rows.")
        return True
    
    if verbose:
        print("No synthetic rows needed to be added.")
    return False