class RateAdjuster:
    def __init__(self, df):
        """
        Initialize the RateAdjuster class with a DataFrame.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe containing measure_id_key, measure_rate, 
            inverse_measure_flag, and lower_rate_is_better_flag
        """
        self.df = df.copy()
        
    def _handle_inversion_flags(self, flag_column):
        """
        Handle the inversion flag columns (inverse_measure_flag or lower_rate_is_better_flag) by:
        1. Counting the number of rows with flag value of 1
        2. If count >= 3, set ALL values (including nulls) to 1
        3. Otherwise, fill null values with 0
        
        Parameters:
        -----------
        flag_column : str
            Name of the flag column to process
        """
        # Count number of rows where flag is 1
        ones_count = (self.df[flag_column] == 1).sum()
        
        # If at least 3 rows have a value of 1
        if ones_count >= 3:
            # Set ALL values to 1 (including nulls)
            self.df[flag_column] = 1
        else:
            # Only if we don't meet the threshold, fill nulls with 0
            self.df[flag_column] = self.df[flag_column].fillna(0)
            
    def _combine_inversion_flags(self):
        """
        Create a combined flag that indicates if the rate should be inverted.
        Rate should be inverted if either inverse_measure_flag or lower_rate_is_better_flag is 1.
        """
        # Process both flag columns
        for flag_column in ['inverse_measure_flag', 'lower_rate_is_better_flag']:
            if flag_column in self.df.columns:
                self._handle_inversion_flags(flag_column)
            else:
                self.df[flag_column] = 0
                
        # Create combined inversion flag
        self.df['should_invert'] = (
            (self.df['inverse_measure_flag'] == 1) | 
            (self.df['lower_rate_is_better_flag'] == 1)
        ).astype(int)
        
    def adjust(self):
        """
        Preprocess the dataframe by:
        1. Converting measure_id_key to object type
        2. Handling both types of inversion flags
        3. Creating combined inversion flag
        4. Adjusting measure_rate based on combined flag
        5. Preserving zero measure rates even for inverted measures
        
        Returns:
        --------
        pandas.DataFrame
            Preprocessed dataframe
        """
        # Convert measure_id_key to object type
        self.df['measure_id_key'] = self.df['measure_id_key'].astype('object')
        
        # Handle and combine inversion flags
        self._combine_inversion_flags()
        
        # Create mask for measures that should be inverted and have non-zero rates
        invert_mask = (self.df['should_invert'] == 1) & (self.df['measure_rate'] != 0)
        
        # Only adjust measure_rate for flagged measures with non-zero rates
        self.df.loc[invert_mask, 'measure_rate'] = 1 - self.df.loc[invert_mask, 'measure_rate']
        
        return self.df

# Example usage:
# df = pd.DataFrame({
#     'measure_id_key': [1, 2, 3],
#     'measure_rate': [0.8, 0.6, 0.4],
#     'inverse_measure_flag': [1, None, 0],
#     'lower_rate_is_better_flag': [0, 1, None]
# })
# adjuster = RateAdjuster(df)
# adjusted_df = adjuster.adjust()