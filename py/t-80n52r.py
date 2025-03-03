# Create a new dataframe by copying 'part0'
fc_df = part0.copy()

# Get the original column list from 'part0' to preserve later
original_columns = part0.columns.tolist()

# Create sets of rows keyed by book_name and measure_id_key in part1
part1_replacement_dict = {}
for _, row in part1.iterrows():
    key = (row['book_name'], row['measure_id_key'])
    part1_replacement_dict[key] = row

# Iterate through the fc_df and replace rows that match part1
for idx, row in fc_df.iterrows():
    key = (row['book_name'], row['measure_id_key'])
    if key in part1_replacement_dict:
        # Replace the row values with values from part1
        fc_df.loc[idx, :] = part1_replacement_dict[key]

# Ensure the columns are in the exact same order as the original part0
fc_df = fc_df[original_columns]