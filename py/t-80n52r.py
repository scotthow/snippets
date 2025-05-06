import pandas as pd

# Assuming hybrid_df already exists
# Step 1: Find the maximum cycle_id and retrieve values
max_cycle_id = hybrid_df['cycle_id'].max()
row_with_max = hybrid_df[hybrid_df['cycle_id'] == max_cycle_id].iloc[0]

# Store the values from the row with max cycle_id
den_max = row_with_max['den']
num_max = row_with_max['num']
measure_rate_max = row_with_max['measure_rate']

# Step 2: Remove rows where cycle_id ends with a number > 16
# Extract the year part from cycle_id (assuming format is YYYY-NN)
hybrid_df['year'] = hybrid_df['cycle_id'].str.split('-').str[0]

# Remove rows where the cycle number is greater than 16
hybrid_df = hybrid_df[~((hybrid_df['cycle_id'].str.split('-').str[1].astype(int) > 16))]

# Step 3: Replace values in the row with cycle_id ending with -16
# First, build the cycle_id with -16 suffix for each year
current_year = max_cycle_id.split('-')[0]  # Get the year from the max cycle_id
target_cycle_id = f"{current_year}-16"

# Replace values in the target row
if target_cycle_id in hybrid_df['cycle_id'].values:
    hybrid_df.loc[hybrid_df['cycle_id'] == target_cycle_id, 'den'] = den_max
    hybrid_df.loc[hybrid_df['cycle_id'] == target_cycle_id, 'num'] = num_max
    hybrid_df.loc[hybrid_df['cycle_id'] == target_cycle_id, 'measure_rate'] = measure_rate_max

# Drop the temporary 'year' column if it's no longer needed
hybrid_df = hybrid_df.drop('year', axis=1)

# Display the modified DataFrame
print(hybrid_df)