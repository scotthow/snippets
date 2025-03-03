import pandas as pd

def apply_datatypes(df1, df2):
    """
    Apply data types from df1 DataFrame to matching columns in df2 DataFrame.
    
    Parameters:
    -----------
    df1 : pandas.DataFrame
        The source DataFrame whose data types will be used as reference
    df2 : pandas.DataFrame
        The target DataFrame whose columns will be converted
        
    Returns:
    --------
    pandas.DataFrame
        A copy of df2 with data types applied from df1 for matching columns
    """
    # Get the data types from df1
    dtypes_dict = df1.dtypes.to_dict()
    
    # Create a copy of df2 to avoid modifying the original
    df2_copy = df2.copy()
    
    # Apply the data types to matching columns in df2
    for column, dtype in dtypes_dict.items():
        if column in df2_copy.columns:
            try:
                # Handle datetime columns specially
                if pd.api.types.is_datetime64_dtype(dtype):
                    df2_copy[column] = pd.to_datetime(df2_copy[column])
                # Handle categorical columns
                elif pd.api.types.is_categorical_dtype(dtype):
                    df2_copy[column] = df2_copy[column].astype('category')
                # For other types
                else:
                    df2_copy[column] = df2_copy[column].astype(dtype)
            except Exception as e:
                print(f"Could not convert column '{column}' to {dtype}: {e}")
    
    return df2_copy

# Example usage:
# df2_converted = apply_datatypes(df1, df2)