# Assuming admin_data and hybrid_data are already loaded

# Creating a composite key for efficient matching
admin_data['composite_key'] = admin_data['book_name'] + '_' + admin_data['measure_id_key'] + '_' + admin_data['cycle_id'].astype(str)
hybrid_data['composite_key'] = hybrid_data['book_name'] + '_' + hybrid_data['measure_id_key'] + '_' + hybrid_data['cycle_id'].astype(str)

# Create a mapping from hybrid_data
hybrid_map = hybrid_data.set_index('composite_key')[['num', 'den', 'measure_rate']].to_dict('index')

# Function to update rows based on the mapping
def update_row(row):
    key = row['composite_key']
    if key in hybrid_map:
        row['num'] = hybrid_map[key]['num']
        row['den'] = hybrid_map[key]['den']
        row['measure_rate'] = hybrid_map[key]['measure_rate']
    return row

# Apply the update function to each row in admin_data
admin_data = admin_data.apply(update_row, axis=1)

# Remove the temporary composite_key column
admin_data = admin_data.drop('composite_key', axis=1)

# Verify the update (optional)
print(f"Updated {len(set(admin_data['composite_key']).intersection(set(hybrid_data['composite_key'])))} records")