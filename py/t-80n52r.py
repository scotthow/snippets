class InverseMeasureRateAdjuster:
    def __init__(self, df):
        """
        Initialize the InverseMeasureRateAdjuster class with a DataFrame.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe containing measure_id_key, measure_rate, and inverse_measure_flag
        """
        self.df = df.copy()
        
    def _handle_inverse_flags(self):
        """
        Handle the inverse_measure_flag column by:
        1. Counting the number of rows with flag value of 1
        2. If count >= 3, set ALL values (including nulls) to 1
        3. Otherwise, fill null values with 0
        """
        # Count number of rows where inverse_measure_flag is 1
        ones_count = (self.df['inverse_measure_flag'] == 1).sum()
        
        # If at least 3 rows have a value of 1
        if ones_count >= 3:
            # Set ALL values to 1 (including nulls)
            self.df['inverse_measure_flag'] = 1
        else:
            # Only if we don't meet the threshold, fill nulls with 0
            self.df['inverse_measure_flag'] = self.df['inverse_measure_flag'].fillna(0)
        
    def adjust(self):
        """
        Preprocess the dataframe by:
        1. Converting measure_id_key to object type
        2. Handling inverse_measure_flag values
        3. Adjusting measure_rate based on inverse_measure_flag
        4. Preserving zero measure rates even for inverted measures
        
        Returns:
        --------
        pandas.DataFrame
            Preprocessed dataframe
        """
        # Convert measure_id_key to object type
        self.df['measure_id_key'] = self.df['measure_id_key'].astype('object')
        
        # Handle inverse measure flags
        self._handle_inverse_flags()
        
        # Create mask for inverse measures that have non-zero rates
        inverse_mask = (self.df['inverse_measure_flag'] == 1) & (self.df['measure_rate'] != 0)
        
        # Only adjust measure_rate for inverse measures with non-zero rates
        self.df.loc[inverse_mask, 'measure_rate'] = 1 - self.df.loc[inverse_mask, 'measure_rate']
        
        return self.df