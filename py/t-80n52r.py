# Create a merged key column in both dataframes for the three matching conditions
admin_data['merge_key'] = admin_data['book_name'] + '_' + admin_data['measure_id_key'].astype(str) + '_' + admin_data['cycle_id'].astype(str)
hybrid_data['merge_key'] = hybrid_data['book_name'] + '_' + hybrid_data['measure_id_key'].astype(str) + '_' + hybrid_data['cycle_id'].astype(str)

# Find all rows in admin_data that have matching keys in hybrid_data
matching_keys = admin_data['merge_key'].isin(hybrid_data['merge_key'])

# For each matching row in admin_data, find the corresponding row in hybrid_data
for key in admin_data.loc[matching_keys, 'merge_key'].unique():
    # Get indices of rows to replace in admin_data
    admin_indices = admin_data[admin_data['merge_key'] == key].index
    
    # Get the replacement data from hybrid_data
    replacement_data = hybrid_data[hybrid_data['merge_key'] == key].iloc[0]
    
    # Replace the rows in admin_data with data from hybrid_data
    for col in admin_data.columns:
        if col != 'merge_key' and col in replacement_data:
            admin_data.loc[admin_indices, col] = replacement_data[col]

# Drop the temporary merge_key column
admin_data = admin_data.drop('merge_key', axis=1)
hybrid_data = hybrid_data.drop('merge_key', axis=1)

# Display the first few rows to confirm
print(admin_data.head())