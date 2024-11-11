class InverseMeasureRateAdjuster:
    def __init__(self, df):
        """
        Initialize the InverseMeasureRateAdjuster class with a DataFrame.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe containing measure_id_key, measure_rate,
            inverse_measure_flag, and lower_rate_is_better_flag
        """
        self.df = df.copy()
    
    def adjust(self):
        """
        Preprocess the dataframe by:
        1. Converting measure_id_key to object type
        2. Adjusting measure_rate based on inversion flags and cycle_id conditions
        
        Returns:
        --------
        pandas.DataFrame
            Preprocessed dataframe
        """
        # Convert measure_id_key to object type
        self.df['measure_id_key'] = self.df['measure_id_key'].astype('object')
        
        # Create mask for measures where cycle_id ends in "16"
        self.df['cycle_id_ends_in_16'] = self.df['cycle_id'].astype(str).str.endswith('16')
        
        # Adjust measure_rate based on the given conditions
        for index, row in self.df.iterrows():
            should_invert = False
            
            # Check if inverse_measure_flag is 1 and cycle_id doesn't end in "16"
            if row['inverse_measure_flag'] == 1 and not row['cycle_id_ends_in_16']:
                should_invert = True
            # Check if lower_rate_is_better_flag is 1 (only if inverse_measure_flag is 0)
            elif row['inverse_measure_flag'] == 0 and row['lower_rate_is_better_flag'] == 1:
                should_invert = True
                
            # Apply inversion if needed
            if should_invert:
                self.df.at[index, 'measure_rate'] = 1 - row['measure_rate']
        
        # Drop the temporary 'cycle_id_ends_in_16' column
        self.df.drop(columns=['cycle_id_ends_in_16'], inplace=True)
        
        return self.df