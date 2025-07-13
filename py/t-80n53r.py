import pandas as pd
import numpy as np


print("--- Original Data Sample ---")
print(df.head())
print(f"\nOriginal data has {df.shape[0]} rows.")


# --- 2. Define Risk Score Columns ---
risk_score_columns = [
    'poverty_score',
    'food_access_score',
    'health_access_score',
    'housing_desert_score',
    'transport_access_score'
]

# --- 3. Calculate Weighted Average Scores by Zip Code ---
# This is the core of the statistical rollup.

# First, we need to create the weighted scores for each member.
# This is simply the score itself, as each row is one member.
# We will sum these up and divide by the count of members per group.
for col in risk_score_columns:
    # The 'weight' is effectively 1 for each member.
    # So, the weighted score is just the score.
    # We create these columns to make the aggregation logic clear.
    df[f'weighted_{col}'] = df[col]

# Group by zip code
grouped = df.groupby('zip_code')

# Count members in each zip code
zip_counts = grouped['src_mbr_id'].count().rename('member_count')

# Sum the weighted scores for each zip code
weighted_sums = grouped[[f'weighted_{col}' for col in risk_score_columns]].sum()

# Combine counts and sums
zip_rollup = pd.concat([zip_counts, weighted_sums], axis=1)

# Calculate the final weighted average for each risk score
for col in risk_score_columns:
    zip_rollup[col] = (zip_rollup[f'weighted_{col}'] / zip_rollup['member_count']).round(2)

# Drop the intermediate weighted sum columns
zip_rollup = zip_rollup.drop(columns=[f'weighted_{col}' for col in risk_score_columns])


print("\n--- Rolled-up Data by Zip Code (Weighted Average) ---")
print(zip_rollup.head())


# --- 4. Normalize Scores into 1-5 Quintile Ranks ---
# We use pd.qcut to bin the scores into 5 equal-sized groups (quintiles).
# Labels are set from 1 (lowest risk) to 5 (highest risk).
for col in risk_score_columns:
    normalized_col_name = f'{col}_rank'
    zip_rollup[normalized_col_name] = pd.qcut(
        zip_rollup[col],
        q=5,
        labels=[1, 2, 3, 4, 5],
        duplicates='drop' # Handles cases where bin edges are not unique
    )

print("\n--- Final Dataframe with Normalized Ranks ---")
# Display the final dataframe with original scores and new ranks
final_df = zip_rollup
print(final_df.head())
print(f"\nFinal dataframe has {final_df.shape[0]} unique zip codes.")

# --- Verification of Normalization ---
print("\n--- Distribution of Ranks for 'poverty_score_rank' ---")
print(final_df['poverty_score_rank'].value_counts().sort_index())


################################### V2 ###################################
# This section is for the second version of the code, which includes additional features or changes.

import pandas as pd
import numpy as np

def rollup_sdoh_to_zipcode(df):
    """
    Roll up SDoH scores from census block level to zip code level
    and create normalized risk categories (1-5).
    
    Parameters:
    df: pandas DataFrame with columns described in the data dictionary
    
    Returns:
    pandas DataFrame with zip code level aggregated and normalized scores
    """
    
    # Step 1: Calculate member counts per block for weighting
    block_member_counts = df.groupby(['zip_code', 'block_code']).size().reset_index(name='member_count')
    
    # Step 2: Get unique block-level scores with member counts
    # Since scores are the same for all members in a block, we can use first()
    block_scores = df.groupby(['zip_code', 'block_code']).agg({
        'poverty_score': 'first',
        'food_access': 'first',
        'health_access': 'first',
        'housing_desert': 'first',
        'transport_access': 'first'
    }).reset_index()
    
    # Merge with member counts
    block_data = block_scores.merge(block_member_counts, on=['zip_code', 'block_code'])
    
    # Step 3: Calculate weighted averages at zip code level
    # Weight by number of members in each block
    risk_columns = ['poverty_score', 'food_access', 'health_access', 'housing_desert', 'transport_access']
    
    # Function to calculate weighted average
    def weighted_avg(group, col):
        return np.average(group[col], weights=group['member_count'])
    
    # Aggregate to zip code level
    zip_aggregated = block_data.groupby('zip_code').apply(
        lambda x: pd.Series({
            col: weighted_avg(x, col) for col in risk_columns
        })
    ).reset_index()
    
    # Add total member count per zip code
    zip_aggregated['total_members'] = block_data.groupby('zip_code')['member_count'].sum().values
    
    # Step 4: Create normalized risk categories (1-5) using quintiles
    # Calculate quintiles for each risk score across all zip codes
    normalized_columns = {}
    
    for col in risk_columns:
        # Calculate quintile thresholds
        quintiles = np.percentile(zip_aggregated[col], [20, 40, 60, 80])
        
        # Create normalized column name
        norm_col = col.replace('_score', '').replace('_access', '').replace('_desert', '') + '_risk_level'
        
        # Assign risk levels based on quintiles
        zip_aggregated[norm_col] = pd.cut(
            zip_aggregated[col],
            bins=[-np.inf] + list(quintiles) + [np.inf],
            labels=[1, 2, 3, 4, 5],
            include_lowest=True
        ).astype(int)
    
    # Reorder columns for clarity
    column_order = ['zip_code', 'total_members'] + risk_columns + [
        'poverty_risk_level', 'food_risk_level', 'health_risk_level', 
        'housing_risk_level', 'transport_risk_level'
    ]
    
    zip_aggregated = zip_aggregated[column_order]
    
    return zip_aggregated

# Apply the function
zip_level_df = rollup_sdoh_to_zipcode(df)

# Display summary statistics
print("Summary of aggregated zip code level data:")
print(f"Number of unique zip codes: {len(zip_level_df)}")
print(f"\nRisk score ranges:")
for col in ['poverty_score', 'food_access', 'health_access', 'housing_desert', 'transport_access']:
    print(f"{col}: {zip_level_df[col].min():.2f} - {zip_level_df[col].max():.2f}")

print(f"\nDistribution of risk levels (should be roughly equal):")
for col in ['poverty_risk_level', 'food_risk_level', 'health_risk_level', 'housing_risk_level', 'transport_risk_level']:
    print(f"\n{col}:")
    print(zip_level_df[col].value_counts().sort_index())