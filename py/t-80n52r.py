import pandas as pd
from google.cloud import bigquery
import sqlalchemy
from sqlalchemy import create_engine
import urllib
import pyodbc

# Configuration
# BigQuery settings
PROJECT_ID = 'your-gcp-project-id'
DATASET_ID = 'your-dataset-id'
TABLE_ID = 'your-table-id'
BIGQUERY_TABLE = f'{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}'

# SQL Server settings
SQL_SERVER_HOST = 'your-sql-server-host'
SQL_SERVER_DATABASE = 'your-database-name'
SQL_SERVER_SCHEMA = 'dbo'  # or your schema name
SQL_SERVER_TABLE = 'your-target-table-name'
SQL_SERVER_USERNAME = 'your-username'
SQL_SERVER_PASSWORD = 'your-password'
SQL_SERVER_DRIVER = 'ODBC Driver 17 for SQL Server'  # or 'ODBC Driver 18 for SQL Server'

# Optional: BigQuery query (if you want to filter/transform data)
# If None, will read entire table
CUSTOM_QUERY = None  # e.g., f"SELECT * FROM `{BIGQUERY_TABLE}` WHERE date > '2024-01-01'"

# Batch size for chunked processing (useful for large tables)
BATCH_SIZE = 10000

def create_bigquery_client():
    """Create and return BigQuery client"""
    return bigquery.Client(project=PROJECT_ID)

def create_sqlserver_connection():
    """Create and return SQL Server connection using SQLAlchemy"""
    # URL encode the password to handle special characters
    params = urllib.parse.quote_plus(
        f'DRIVER={{{SQL_SERVER_DRIVER}}};'
        f'SERVER={SQL_SERVER_HOST};'
        f'DATABASE={SQL_SERVER_DATABASE};'
        f'UID={SQL_SERVER_USERNAME};'
        f'PWD={SQL_SERVER_PASSWORD};'
        'TrustServerCertificate=yes;'  # Add if using self-signed certificates
    )
    
    connection_string = f'mssql+pyodbc:///?odbc_connect={params}'
    engine = create_engine(connection_string, fast_executemany=True)
    
    return engine

def read_from_bigquery(client, query=None):
    """
    Read data from BigQuery table or custom query
    
    Args:
        client: BigQuery client
        query: Optional custom SQL query
    
    Returns:
        pandas DataFrame
    """
    if query:
        print(f"Executing custom query...")
        df = client.query(query).to_dataframe()
    else:
        print(f"Reading entire table: {BIGQUERY_TABLE}")
        df = client.query(f"SELECT * FROM `{BIGQUERY_TABLE}`").to_dataframe()
    
    print(f"Retrieved {len(df)} rows from BigQuery")
    return df

def read_from_bigquery_chunked(client, query=None, chunk_size=BATCH_SIZE):
    """
    Read data from BigQuery in chunks (for large tables)
    
    Args:
        client: BigQuery client
        query: Optional custom SQL query
        chunk_size: Number of rows per chunk
    
    Yields:
        pandas DataFrame chunks
    """
    if query:
        query_job = client.query(query)
    else:
        query_job = client.query(f"SELECT * FROM `{BIGQUERY_TABLE}`")
    
    # Get total row count
    total_rows = query_job.result().total_rows
    print(f"Total rows to process: {total_rows}")
    
    # Process in chunks
    offset = 0
    while offset < total_rows:
        chunk_query = f"{query or f'SELECT * FROM `{BIGQUERY_TABLE}`'} LIMIT {chunk_size} OFFSET {offset}"
        chunk_df = client.query(chunk_query).to_dataframe()
        
        if chunk_df.empty:
            break
            
        print(f"Processing rows {offset} to {offset + len(chunk_df)}")
        yield chunk_df
        
        offset += chunk_size

def write_to_sqlserver(df, engine, table_name, schema=SQL_SERVER_SCHEMA, if_exists='replace'):
    """
    Write DataFrame to SQL Server
    
    Args:
        df: pandas DataFrame
        engine: SQLAlchemy engine
        table_name: Target table name
        schema: Database schema
        if_exists: 'replace', 'append', or 'fail'
    """
    try:
        df.to_sql(
            name=table_name,
            con=engine,
            schema=schema,
            if_exists=if_exists,
            index=False,
            method='multi',  # Faster insertion
            chunksize=1000   # Insert in batches
        )
        print(f"Successfully wrote {len(df)} rows to {schema}.{table_name}")
    except Exception as e:
        print(f"Error writing to SQL Server: {str(e)}")
        raise

def transfer_data_simple():
    """Simple transfer for small to medium tables"""
    print("Starting data transfer from BigQuery to SQL Server...")
    
    # Create connections
    bq_client = create_bigquery_client()
    sql_engine = create_sqlserver_connection()
    
    try:
        # Read from BigQuery
        df = read_from_bigquery(bq_client, CUSTOM_QUERY)
        
        # Display sample data
        print("\nSample data (first 5 rows):")
        print(df.head())
        
        # Write to SQL Server
        write_to_sqlserver(df, sql_engine, SQL_SERVER_TABLE)
        
        print("\nData transfer completed successfully!")
        
    except Exception as e:
        print(f"Error during transfer: {str(e)}")
        raise
    finally:
        sql_engine.dispose()

def transfer_data_chunked():
    """Chunked transfer for large tables"""
    print("Starting chunked data transfer from BigQuery to SQL Server...")
    
    # Create connections
    bq_client = create_bigquery_client()
    sql_engine = create_sqlserver_connection()
    
    try:
        first_chunk = True
        total_transferred = 0
        
        # Process data in chunks
        for chunk_df in read_from_bigquery_chunked(bq_client, CUSTOM_QUERY):
            # Write to SQL Server
            # First chunk replaces table, subsequent chunks append
            if_exists = 'replace' if first_chunk else 'append'
            write_to_sqlserver(chunk_df, sql_engine, SQL_SERVER_TABLE, if_exists=if_exists)
            
            total_transferred += len(chunk_df)
            first_chunk = False
        
        print(f"\nData transfer completed! Total rows transferred: {total_transferred}")
        
    except Exception as e:
        print(f"Error during transfer: {str(e)}")
        raise
    finally:
        sql_engine.dispose()

def verify_transfer():
    """Verify the data transfer by comparing row counts"""
    print("\nVerifying data transfer...")
    
    # Count rows in BigQuery
    bq_client = create_bigquery_client()
    bq_count_query = f"SELECT COUNT(*) as count FROM `{BIGQUERY_TABLE}`"
    bq_count = bq_client.query(bq_count_query).to_dataframe()['count'][0]
    print(f"BigQuery row count: {bq_count}")
    
    # Count rows in SQL Server
    sql_engine = create_sqlserver_connection()
    try:
        sql_count_query = f"SELECT COUNT(*) as count FROM {SQL_SERVER_SCHEMA}.{SQL_SERVER_TABLE}"
        sql_count = pd.read_sql(sql_count_query, sql_engine)['count'][0]
        print(f"SQL Server row count: {sql_count}")
        
        if bq_count == sql_count:
            print("✓ Row counts match! Transfer verified.")
        else:
            print(f"✗ Row count mismatch! Difference: {bq_count - sql_count}")
    finally:
        sql_engine.dispose()

# Main execution
if __name__ == "__main__":
    # Choose transfer method based on your table size
    # For small to medium tables (< 1 million rows)
    transfer_data_simple()
    
    # For large tables (> 1 million rows)
    # transfer_data_chunked()
    
    # Verify the transfer
    verify_transfer()

# Example: Transfer with custom data types mapping
def transfer_with_dtype_mapping():
    """Transfer with explicit data type mapping"""
    from sqlalchemy.types import Integer, String, Float, DateTime, Boolean
    
    bq_client = create_bigquery_client()
    sql_engine = create_sqlserver_connection()
    
    try:
        df = read_from_bigquery(bq_client, CUSTOM_QUERY)
        
        # Define SQL Server data types for specific columns
        dtype_mapping = {
            'id': Integer(),
            'name': String(255),
            'amount': Float(),
            'created_date': DateTime(),
            'is_active': Boolean()
        }
        
        df.to_sql(
            name=SQL_SERVER_TABLE,
            con=sql_engine,
            schema=SQL_SERVER_SCHEMA,
            if_exists='replace',
            index=False,
            dtype=dtype_mapping  # Apply custom data types
        )
        
        print("Data transferred with custom data types!")
        
    finally:
        sql_engine.dispose()

# Example: Incremental transfer based on timestamp
def incremental_transfer(timestamp_column='updated_at', last_sync_time='2024-01-01 00:00:00'):
    """Transfer only new/updated records based on timestamp"""
    
    incremental_query = f"""
    SELECT * FROM `{BIGQUERY_TABLE}`
    WHERE {timestamp_column} > '{last_sync_time}'
    """
    
    bq_client = create_bigquery_client()
    sql_engine = create_sqlserver_connection()
    
    try:
        df = read_from_bigquery(bq_client, incremental_query)
        
        if not df.empty:
            write_to_sqlserver(df, sql_engine, SQL_SERVER_TABLE, if_exists='append')
            print(f"Incremental transfer completed: {len(df)} new/updated rows")
        else:
            print("No new records to transfer")
            
    finally:
        sql_engine.dispose()