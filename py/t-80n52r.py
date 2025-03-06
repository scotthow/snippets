def update_dataframe_values(df1, df2):
    """
    Replace 'num', 'den', and 'measure_rate' values in df1 if 'book_name' and 'measure_id_key'
    match with those in df2 (hybrid dataframe).
    
    Parameters:
    df1 (pandas.DataFrame): The original dataframe to be updated
    df2 (pandas.DataFrame): The hybrid dataframe containing new values
    
    Returns:
    pandas.DataFrame: The updated dataframe
    """
    # Create a copy of the original dataframe to avoid modifying it directly
    updated_df = df1.copy()
    
    # Create a mapping for efficient lookups
    hybrid_mapping = {}
    for _, row in df2.iterrows():
        key = (row['book_name'], row['measure_id_key'])
        hybrid_mapping[key] = {
            'num': row['num'],
            'den': row['den'],
            'measure_rate': row['measure_rate']
        }
    
    # Update the df1 where matches are found
    for idx, row in updated_df.iterrows():
        key = (row['book_name'], row['measure_id_key'])
        if key in hybrid_mapping:
            updated_df.at[idx, 'num'] = hybrid_mapping[key]['num']
            updated_df.at[idx, 'den'] = hybrid_mapping[key]['den']
            updated_df.at[idx, 'measure_rate'] = hybrid_mapping[key]['measure_rate']
    
    return updated_df



import pandas as pd

# Assuming you have a DataFrame loaded from the SQL database
# df = pd.read_sql("SELECT * FROM HEDIS.MHJ.QPR_IFP_HYBRID_HIST", connection)

# Get the maximum Run_Month value
max_run_month = df['Run_Month'].max()

# Filter the data
filtered_df = df[
    (df['Run_Month'] == max_run_month) & 
    (df['Max_Month'] == 1) & 
    (df['LOB'] == 'IFP')
]

# Create the calculated field for 'num'
def calculate_num(row):
    if row['Measure_ID_Key'] in ['100501', '100201', '105201']:
        return row['MRSS_Denom'] - (row['MRSS_Admin_Num'] + row['MRSS_Med_Num'])
    else:
        return row['MRSS_Admin_Num'] + row['MRSS_Med_Num']

# Apply the calculation
filtered_df['num'] = filtered_df.apply(calculate_num, axis=1)

# Select and rename columns
result_df = filtered_df[[
    'measurement_year',
    'Run_Month',
    'Book_Name',
    'Internal_QPR_Measure_Key',
    'Measure_ID_Key',
    'hybrid_time',
    'measure_name',
    'MRSS_Denom',
    'num',
    'measure_rate'
]].rename(columns={
    'Run_Month': 'cycle_id',
    'MRSS_Denom': 'den'
})

# Sort the results
result_df = result_df.sort_values(by=['Book_Name', 'Measure_ID_Key'])