# Helper function to safely get rate value
def get_rate_value(df, cycle_id):
    matching_rows = df[df['cycle_id'] == cycle_id]['measure_rate']
    return matching_rows.values[0] if not matching_rows.empty else None

# Current rate this month last year
last_year_cycle_id = f"{str(int(curr_cycle_id[:4]) - 1)}{curr_cycle_id[4:]}"
last_year_rate = get_rate_value(df, last_year_cycle_id)
print(f"Current cycle id: {curr_cycle_id}, Last year cycle id: {last_year_cycle_id}")

# Define cycle IDs and get their rates
cycles = {
    'my22_last_admin': '2022-15',
    'my22_reported': '2022-16',
    'my23_last_admin': '2023-15',
    'my23_reported': '2023-16'
}

# Get rates using dictionary comprehension
rates = {f"{key}_rate": get_rate_value(df, cycle_id) 
         for key, cycle_id in cycles.items()}

# Unpack the rates into individual variables
my22_last_admin_rate = rates['my22_last_admin_rate']
my22_reported_rate = rates['my22_reported_rate']
my23_last_admin_rate = rates['my23_last_admin_rate']
my23_reported_rate = rates['my23_reported_rate']