import os
import pandas as pd
from pathlib import Path
import glob

def list_parquet_files(directory_path):
    """List all parquet files in the specified directory."""
    # Convert to Path object for better cross-platform compatibility
    dir_path = Path(directory_path)

    # Find all parquet files
    parquet_files = list(dir_path.glob("*.parquet"))

    if not parquet_files:
        print(f"No parquet files found in {directory_path}")
        return []

    print(f"Found {len(parquet_files)} parquet files:")
    for file in sorted(parquet_files):
        print(f"  - {file.name}")

    return parquet_files

def read_episode_parquet(file_path):
    """Read a single parquet file and return the DataFrame."""
    try:
        df = pd.read_parquet(file_path)
        print(f"\nSuccessfully read {file_path.name}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"All rows:")

        # Set pandas display options to show all rows and columns
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)

        print(df.to_string())

        # Reset pandas display options to default
        pd.reset_option('display.max_rows')
        pd.reset_option('display.max_columns')
        pd.reset_option('display.width')
        pd.reset_option('display.max_colwidth')

        return df
    except Exception as e:
        print(f"Error reading {file_path.name}: {e}")
        return None

def read_specific_episode(directory_path, episode_number):
    """Read a specific episode by number."""
    episode_file = f"episode_{episode_number:06d}.parquet"
    file_path = Path(directory_path) / episode_file

    if file_path.exists():
        return read_episode_parquet(file_path)
    else:
        print(f"Episode file {episode_file} not found")
        return None

def main():
    # Your directory path
    directory = r"C:\Users\Nicholas\.cache\huggingface\lerobot\local\sourccey-002__stiction_tape-test-a__set001__chrism\data\chunk-000"

    print("=== Parquet File Reader ===")
    print(f"Directory: {directory}")
    print()

    # List all available parquet files
    parquet_files = list_parquet_files(directory)

    if not parquet_files:
        return

    print("\n" + "="*50)

    # Read just episode 0
    episode_num = 0
    print(f"\n--- Reading Episode {episode_num:06d} ---")
    df = read_specific_episode(directory, episode_num)

    if df is not None:
        # You can add more analysis here
        print(f"Episode {episode_num} data loaded successfully!")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

    print("\n" + "="*50)
    print("Script completed!")

if __name__ == "__main__":
    main()
