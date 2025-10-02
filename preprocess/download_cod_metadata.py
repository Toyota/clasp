import argparse
import pandas as pd
import mysql.connector

def download_cod_metadata(output_filename: str):
    """
    Download COD metadata and save it as a CSV file.

    Args:
        output_filename (str): The name of the output file to save the CSV data.
    """
    config = {
        'user': 'cod_reader',
        'password': '',  # not needed
        'host': 'sql.crystallography.net',
        'database': 'cod'}

    # connect to COD
    connection = mysql.connector.connect(**config)

    # Retrieve all metadata and make dataframe
    query = "SELECT * FROM data"
    result_df = pd.read_sql_query(query, connection)

    # Save the result as a CSV file
    result_df.to_csv(output_filename, index=False)

if __name__ == "__main__":
    """
    Example:
    python download_cod_metadata.py cod_metadata_20230523.csv
    """
    parser = argparse.ArgumentParser(description="Download COD metadata and save it as a CSV file.")
    parser.add_argument("output_filename", type=str, help="The name of the output file to save the CSV data.")
    args = parser.parse_args()

    download_cod_metadata(args.output_filename)
    print(f"file saved: {args.output_filename}")
    print("complete")