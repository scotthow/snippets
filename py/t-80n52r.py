import pandas as pd
import os
from sqlalchemy import create_engine
import urllib

# Define the directory and index columns
directory = 'Analytics_Temp/mcaid'
index_cols = ['air_pollution_index', 'education_index', 'food_access', 'health_access']

# Initialize list to store all dataframes
all_dataframes = []

# Process each file in the directory
for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
    
    # Skip if not a file
    if not os.path.isfile(filepath):
        continue
    
    try:
        # Read file into dataframe (handles CSV, Excel, etc.)
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(filepath)
        else:
            print(f"Skipping unsupported file: {filename}")
            continue
        
        # Calculate zip code averages for each index column
        for col in index_cols:
            if col in df.columns:
                # Group by zip_code and calculate mean
                zip_avg = df.groupby('zip_code')[col].mean().reset_index()
                zip_avg.columns = ['zip_code', f'{col}_zip']
                
                # Merge back to original dataframe
                df = df.merge(zip_avg, on='zip_code', how='left')
        
        # Add to list of dataframes
        all_dataframes.append(df)
        print(f"Processed: {filename}")
        
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")

# Combine all dataframes into one
if all_dataframes:
    main_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"\nCombined {len(all_dataframes)} files into main dataframe")
    print(f"Total rows: {len(main_df)}")
else:
    print("No dataframes to combine")
    main_df = pd.DataFrame()

# Upload to SQL Server with trusted connection
if not main_df.empty:
    # Create connection string for Windows Authentication
    server = 'YOUR_SERVER_NAME'  # Replace with your server name
    database = 'YOUR_DATABASE_NAME'  # Replace with your database name
    table_name = 'YOUR_TABLE_NAME'  # Replace with your desired table name
    
    # Connection string for trusted connection (Windows Authentication)
    params = urllib.parse.quote_plus(
        f'DRIVER={{ODBC Driver 17 for SQL Server}};'
        f'SERVER={server};'
        f'DATABASE={database};'
        f'Trusted_Connection=yes;'
    )
    
    # Create engine
    engine = create_engine(f'mssql+pyodbc:///?odbc_connect={params}')
    
    try:
        # Upload dataframe to SQL Server
        main_df.to_sql(
            name=table_name,
            con=engine,
            if_exists='replace',  # or 'append' if you want to add to existing table
            index=False
        )
        print(f"\nSuccessfully uploaded to SQL Server table: {table_name}")
    except Exception as e:
        print(f"Error uploading to SQL Server: {str(e)}")