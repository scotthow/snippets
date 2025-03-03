import pandas as pd

def apply_datatypes(part1, part2):
    """
    Apply data types from part1 DataFrame to matching columns in part2 DataFrame.
    
    Parameters:
    -----------
    part1 : pandas.DataFrame
        The source DataFrame whose data types will be used as reference
    part2 : pandas.DataFrame
        The target DataFrame whose columns will be converted
        
    Returns:
    --------
    pandas.DataFrame
        A copy of part2 with data types applied from part1 for matching columns
    """
    # Get the data types from part1
    dtypes_dict = part1.dtypes.to_dict()
    
    # Create a copy of part2 to avoid modifying the original
    part2_copy = part2.copy()
    
    # Apply the data types to matching columns in part2
    for column, dtype in dtypes_dict.items():
        if column in part2_copy.columns:
            try:
                # Handle datetime columns specially
                if pd.api.types.is_datetime64_dtype(dtype):
                    part2_copy[column] = pd.to_datetime(part2_copy[column])
                # Handle categorical columns
                elif pd.api.types.is_categorical_dtype(dtype):
                    part2_copy[column] = part2_copy[column].astype('category')
                # For other types
                else:
                    part2_copy[column] = part2_copy[column].astype(dtype)
            except Exception as e:
                print(f"Could not convert column '{column}' to {dtype}: {e}")
    
    return part2_copy

# Example usage:
# part2_converted = apply_datatypes(part1, part2)