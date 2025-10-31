#!/usr/bin/env python3
"""
Script to view parquet data from LeRobot datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, Any, List
import argparse


def load_parquet_file(file_path: str) -> pd.DataFrame:
    """Load a parquet file and return as pandas DataFrame."""
    try:
        df = pd.read_parquet(file_path)
        return df
    except Exception as e:
        print(f"Error loading parquet file: {e}")
        return None


def display_basic_info(df: pd.DataFrame) -> None:
    """Display basic information about the DataFrame."""
    print("=" * 80)
    print("BASIC INFORMATION")
    print("=" * 80)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types:")
    for col, dtype in df.dtypes.items():
        print(f"  {col}: {dtype}")
    print()


def display_sample_data(df: pd.DataFrame, n_rows: int = 5) -> None:
    """Display sample rows from the DataFrame."""
    print("=" * 80)
    print(f"SAMPLE DATA (first {n_rows} rows)")
    print("=" * 80)
    print(df.head(n_rows))
    print()


def display_statistics(df: pd.DataFrame) -> None:
    """Display statistical information for numeric columns."""
    print("=" * 80)
    print("STATISTICAL SUMMARY")
    print("=" * 80)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print("Numeric columns statistics:")
        print(df[numeric_cols].describe())
        print()
    
    # Check for any columns that might contain arrays/tensors
    array_cols = []
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if first non-null value is an array-like
            first_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
            if first_val is not None and hasattr(first_val, '__len__') and not isinstance(first_val, str):
                array_cols.append(col)
    
    if array_cols:
        print("Array-like columns:")
        for col in array_cols:
            first_val = df[col].dropna().iloc[0]
            print(f"  {col}: shape {getattr(first_val, 'shape', len(first_val))}, dtype {type(first_val)}")
        print()


def display_column_details(df: pd.DataFrame) -> None:
    """Display detailed information about each column."""
    print("=" * 80)
    print("COLUMN DETAILS")
    print("=" * 80)
    
    for col in df.columns:
        print(f"\nColumn: {col}")
        print(f"  Data type: {df[col].dtype}")
        print(f"  Non-null count: {df[col].count()}")
        print(f"  Null count: {df[col].isnull().sum()}")
        
        if df[col].dtype == 'object':
            # For object columns, show unique values or sample values
            unique_vals = df[col].nunique()
            print(f"  Unique values: {unique_vals}")
            
            if unique_vals <= 10:
                print(f"  Values: {df[col].unique()}")
            else:
                print(f"  Sample values: {df[col].dropna().head(3).tolist()}")
        else:
            # For numeric columns, show range
            if not df[col].isnull().all():
                print(f"  Min: {df[col].min()}")
                print(f"  Max: {df[col].max()}")


def explore_nested_data(df: pd.DataFrame) -> None:
    """Explore nested data structures in the DataFrame."""
    print("=" * 80)
    print("NESTED DATA EXPLORATION")
    print("=" * 80)
    
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if this might be a nested structure
            sample_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
            
            if sample_val is not None:
                if isinstance(sample_val, dict):
                    print(f"\nColumn '{col}' contains dictionaries:")
                    print(f"  Keys: {list(sample_val.keys())}")
                    for key, value in sample_val.items():
                        print(f"    {key}: {type(value)} - {str(value)[:100]}...")
                
                elif hasattr(sample_val, '__len__') and not isinstance(sample_val, str):
                    print(f"\nColumn '{col}' contains arrays/tensors:")
                    print(f"  Type: {type(sample_val)}")
                    print(f"  Shape: {getattr(sample_val, 'shape', len(sample_val))}")
                    if hasattr(sample_val, 'dtype'):
                        print(f"  Dtype: {sample_val.dtype}")


def main():
    parser = argparse.ArgumentParser(description="View parquet data from LeRobot datasets")
    parser.add_argument("file_path", help="Path to the parquet file")
    parser.add_argument("--rows", type=int, default=5, help="Number of sample rows to display")
    parser.add_argument("--detailed", action="store_true", help="Show detailed column information")
    parser.add_argument("--nested", action="store_true", help="Explore nested data structures")
    
    args = parser.parse_args()
    
    file_path = Path(args.file_path)
    if not file_path.exists():
        print(f"Error: File {file_path} does not exist")
        return
    
    print(f"Loading parquet file: {file_path}")
    df = load_parquet_file(str(file_path))
    
    if df is None:
        return
    
    # Display basic information
    display_basic_info(df)
    
    # Display sample data
    display_sample_data(df, args.rows)
    
    # Display statistics
    display_statistics(df)
    
    # Display detailed column information if requested
    if args.detailed:
        display_column_details(df)
    
    # Explore nested data if requested
    if args.nested:
        explore_nested_data(df)
    
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
