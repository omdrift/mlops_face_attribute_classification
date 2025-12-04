"""
Script to merge multiple annotation CSV files from data/labels_raw/ into a single CSV.

This script:
1. Scans data/labels_raw/ for all CSV files
2. Merges them into a single CSV file
3. Removes duplicates based on filename
4. Saves the result to data/annotations/merged_labels.csv

Usage:
    python merge_annotations.py
    python merge_annotations.py --input-dir data/labels_raw --output data/annotations/merged_labels.csv
"""
import os
import argparse
import pandas as pd
from pathlib import Path
from typing import List


def find_csv_files(input_dir: str) -> List[Path]:
    """
    Find all CSV files recursively in the input directory.
    
    Args:
        input_dir: Directory to search for CSV files
        
    Returns:
        List of paths to CSV files found
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"[!] Input directory does not exist: {input_dir}")
        return []
    
    csv_files = list(input_path.rglob("*.csv"))
    
    if not csv_files:
        print(f"[!] No CSV files found in {input_dir}")
        return []
    
    print(f"[*] Found {len(csv_files)} CSV file(s) in {input_dir}")
    for csv_file in sorted(csv_files):
        relative_path = csv_file.relative_to(input_path)
        print(f"    - {relative_path}")
    
    return csv_files


def merge_csv_files(csv_files: List[Path]) -> pd.DataFrame:
    """
    Merge multiple CSV files into a single DataFrame.
    
    Args:
        csv_files: List of paths to CSV files
        
    Returns:
        Merged DataFrame
    """
    if not csv_files:
        return pd.DataFrame()
    
    print(f"\n[*] Merging {len(csv_files)} CSV file(s)...")
    
    dataframes = []
    total_rows = 0
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            rows = len(df)
            total_rows += rows
            dataframes.append(df)
            print(f"    - {csv_file.name}: {rows} rows")
        except Exception as e:
            print(f"    [!] Error reading {csv_file.name}: {e}")
            continue
    
    if not dataframes:
        print("[!] No valid CSV files to merge")
        return pd.DataFrame()
    
    # Concatenate all dataframes
    merged_df = pd.concat(dataframes, ignore_index=True)
    print(f"\n[+] Total rows before deduplication: {total_rows}")
    
    return merged_df


def remove_duplicates(df: pd.DataFrame, key_column: str = 'filename') -> pd.DataFrame:
    """
    Remove duplicate rows based on a key column.
    
    Args:
        df: Input DataFrame
        key_column: Column to use for deduplication (default: 'filename')
        
    Returns:
        DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    if key_column not in df.columns:
        print(f"[!] Warning: Column '{key_column}' not found. Available columns: {df.columns.tolist()}")
        return df
    
    initial_count = len(df)
    
    # Keep last occurrence (most recent annotation)
    df_dedup = df.drop_duplicates(subset=[key_column], keep='last')
    
    duplicates_removed = initial_count - len(df_dedup)
    
    print(f"[*] Removed {duplicates_removed} duplicate(s)")
    print(f"[+] Final row count: {len(df_dedup)}")
    
    return df_dedup


def save_merged_csv(df: pd.DataFrame, output_path: str):
    """
    Save the merged DataFrame to a CSV file.
    
    Args:
        df: DataFrame to save
        output_path: Path where to save the CSV
    """
    if df.empty:
        print("[!] No data to save")
        return
    
    # Create output directory if it doesn't exist
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print(f"\n[+] Merged CSV saved to: {output_path}")
    print(f"    Rows: {len(df)}")
    print(f"    Columns: {df.columns.tolist()}")
    
    # Show sample statistics if label columns exist
    label_cols = ['beard', 'mustache', 'glasses_binary', 'hair_color_label', 'hair_length']
    available_labels = [col for col in label_cols if col in df.columns]
    
    if available_labels:
        print(f"\n[*] Label distribution:")
        for col in available_labels:
            value_counts = df[col].value_counts().to_dict()
            print(f"    {col}: {value_counts}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple annotation CSV files into one"
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/labels_raw',
        help='Directory containing CSV files to merge (default: data/labels_raw)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/annotations/merged_labels.csv',
        help='Output path for merged CSV (default: data/annotations/merged_labels.csv)'
    )
    parser.add_argument(
        '--key-column',
        type=str,
        default='filename',
        help='Column to use for deduplication (default: filename)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(" MERGE ANNOTATION CSV FILES")
    print("=" * 60)
    print(f" Input directory: {args.input_dir}")
    print(f" Output file: {args.output}")
    print(f" Dedup column: {args.key_column}")
    print("=" * 60)
    
    # Find all CSV files
    csv_files = find_csv_files(args.input_dir)
    
    if not csv_files:
        print("\n[!] No CSV files to merge. Exiting.")
        return
    
    # Merge CSV files
    merged_df = merge_csv_files(csv_files)
    
    if merged_df.empty:
        print("\n[!] Merged DataFrame is empty. Exiting.")
        return
    
    # Remove duplicates
    dedup_df = remove_duplicates(merged_df, args.key_column)
    
    # Save result
    save_merged_csv(dedup_df, args.output)
    
    print("\n" + "=" * 60)
    print(" MERGE COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
