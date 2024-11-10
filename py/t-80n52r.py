def handle_missing_early_cycles(self, target_measure_ids=None, verbose=True):
    """
    Handles missing early cycles (01-03) for specific measure IDs in 2023.
    Creates synthetic rows based on the first available cycle (04) data.
    
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
    
    # Convert index levels to integers to ensure consistent typing
    self.df.index = self.df.index.set_levels([
        self.df.index.levels[0].astype(np.int64),
        self.df.index.levels[1].astype(np.int64)
    ])
    
    # Find the first row for 2023-04
    source_idx = (2023, 4)
    if source_idx not in self.df.index:
        if verbose:
            print(f"Cycle 2023-04 not found for measure ID {current_measure_id}. "
                  f"Cannot create synthetic rows. DataFrame remains unchanged.")
        return False
        
    source_row = self.df.loc[source_idx].copy()
    
    # Determine if we should add or subtract based on flags
    inverse_logic = (
        source_row['inverse_measure_flag'] == 1 or 
        source_row['lower_rate_is_better_flag'] == 1
    )
    
    # Create synthetic rows for periods 1-3
    new_rows = []
    base_rate = source_row['measure_rate']
    original_row_count = len(self.df)
    
    for period in range(3, 0, -1):
        # Skip if row already exists
        if (2023, period) in self.df.index:
            if verbose:
                print(f"Cycle 2023-{period:02d} already exists. Skipping.")
            continue
            
        # Create new row
        new_row = source_row.copy()
        
        # Calculate new measure rate
        steps_from_04 = 4 - period  # Number of steps back from period 04
        if inverse_logic:
            adjustment = 0.01 * steps_from_04  # Add 1 percentage point per step
        else:
            adjustment = -0.01 * steps_from_04  # Subtract 1 percentage point per step
        
        new_row['measure_rate'] = base_rate + adjustment
        new_row['synthetic'] = 1
        new_row['instance_data_through_dt'] = None
        
        # Create new index with correct integer types
        new_idx = pd.MultiIndex.from_tuples(
            [(np.int64(2023), np.int64(period))], 
            names=['year', 'period']
        )
        
        # Add to new rows list
        new_df = pd.DataFrame([new_row])
        new_df.index = new_idx
        new_rows.append(new_df)
        
        if verbose:
            print(f"Created synthetic row for cycle 2023-{period:02d} with "
                  f"measure rate {new_row['measure_rate']:.4f}")
    
    # Add all new rows to DataFrame
    if new_rows:
        self.df = pd.concat([self.df] + new_rows)
        
        # Ensure index types are consistent
        self.df.index = self.df.index.set_levels([
            self.df.index.levels[0].astype(np.int64),
            self.df.index.levels[1].astype(np.int64)
        ])
        
        # Sort index to maintain proper order
        self.df.sort_index(inplace=True)
    
    if verbose:
        rows_added = len(self.df) - original_row_count
        print(f"Processing complete. Added {rows_added} synthetic rows.")
    
    return True