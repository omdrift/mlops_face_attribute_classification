"""
Script to map/transform merged annotations into the final training CSV format.

This script:
1. Reads the merged_labels.csv file
2. Validates and cleans the data
3. Maps column names to the expected format
4. Filters out invalid entries
5. Saves the result to mapped_train.csv

Usage:
    python map_annotations.py
    python map_annotations.py --input data/annotations/merged_labels.csv --output data/annotations/mapped_train.csv
"""
import os
import argparse
import pandas as pd
from pathlib import Path


# Label value constraints
BINARY_LABELS = ['beard', 'mustache', 'glasses_binary']
BINARY_VALUES = [0, 1]
HAIR_COLOR_VALUES = [0, 1, 2, 3, 4]
HAIR_LENGTH_VALUES = [0, 1, 2]
REQUIRED_COLUMNS = ['filename', 'beard', 'mustache', 'glasses_binary', 
                   'hair_color_label', 'hair_length']


def load_merged_labels(input_path: str) -> pd.DataFrame:
    """
    Load the merged labels CSV file.
    
    Args:
        input_path: Path to merged_labels.csv
        
    Returns:
        DataFrame with merged labels
    """
    if not os.path.exists(input_path):
        print(f"[!] Input file not found: {input_path}")
        return pd.DataFrame()
    
    print(f"[*] Loading merged labels from: {input_path}")
    
    df = pd.read_csv(input_path)
    
    print(f"    Rows: {len(df)}")
    print(f"    Columns: {df.columns.tolist()}")
    
    return df


def clean_filenames(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean filename column by removing common issues.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with cleaned filenames
    """
    if df.empty or 'filename' not in df.columns:
        return df
    
    print("\n[*] Cleaning filenames...")
    
    initial_count = len(df)
    
    # Remove .csv.png artifacts (if any)
    df['filename'] = df['filename'].str.replace('.csv.png', '.png', regex=False)
    
    # Remove any leading/trailing whitespace
    df['filename'] = df['filename'].str.strip()
    
    # Remove rows with empty filenames
    df = df[df['filename'].notna() & (df['filename'] != '')]
    
    rows_removed = initial_count - len(df)
    
    if rows_removed > 0:
        print(f"    [!] Removed {rows_removed} row(s) with invalid filenames")
    
    print(f"    [+] Valid filenames: {len(df)}")
    
    return df


def validate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate that required label columns exist and contain valid values.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with validated labels
    """
    if df.empty:
        return df
    
    print("\n[*] Validating labels...")
    
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    
    if missing_columns:
        print(f"    [!] Missing required columns: {missing_columns}")
        return pd.DataFrame()
    
    initial_count = len(df)
    
    # Remove rows with NaN in any label column
    df = df.dropna(subset=REQUIRED_COLUMNS)
    
    rows_removed = initial_count - len(df)
    
    if rows_removed > 0:
        print(f"    [!] Removed {rows_removed} row(s) with missing labels")
    
    # Validate binary labels (beard, mustache, glasses_binary)
    for col in BINARY_LABELS:
        invalid_mask = ~df[col].isin(BINARY_VALUES)
        if invalid_mask.any():
            invalid_count = invalid_mask.sum()
            print(f"    [!] Found {invalid_count} invalid value(s) in '{col}' (not in {BINARY_VALUES})")
            df = df[~invalid_mask]
    
    # Validate hair_color_label
    invalid_hair_color = ~df['hair_color_label'].isin(HAIR_COLOR_VALUES)
    if invalid_hair_color.any():
        invalid_count = invalid_hair_color.sum()
        print(f"    [!] Found {invalid_count} invalid value(s) in 'hair_color_label' (not in {HAIR_COLOR_VALUES})")
        df = df[~invalid_hair_color]
    
    # Validate hair_length
    invalid_hair_length = ~df['hair_length'].isin(HAIR_LENGTH_VALUES)
    if invalid_hair_length.any():
        invalid_count = invalid_hair_length.sum()
        print(f"    [!] Found {invalid_count} invalid value(s) in 'hair_length' (not in {HAIR_LENGTH_VALUES})")
        df = df[~invalid_hair_length]
    
    print(f"    [+] Valid labels: {len(df)}")
    
    return df


def map_to_training_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map the DataFrame to the expected training format.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame in training format
    """
    if df.empty:
        return df
    
    print("\n[*] Mapping to training format...")
    
    # Select and reorder columns
    df_mapped = df[REQUIRED_COLUMNS].copy()
    
    # Convert to appropriate types
    df_mapped['beard'] = df_mapped['beard'].astype(int)
    df_mapped['mustache'] = df_mapped['mustache'].astype(int)
    df_mapped['glasses_binary'] = df_mapped['glasses_binary'].astype(int)
    df_mapped['hair_color_label'] = df_mapped['hair_color_label'].astype(int)
    df_mapped['hair_length'] = df_mapped['hair_length'].astype(int)
    
    print(f"    [+] Mapped {len(df_mapped)} rows")
    
    return df_mapped


def save_mapped_labels(df: pd.DataFrame, output_path: str):
    """
    Save the mapped labels to the training CSV file.
    
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
    
    print(f"\n[+] Mapped training CSV saved to: {output_path}")
    print(f"    Rows: {len(df)}")
    print(f"    Columns: {df.columns.tolist()}")
    
    # Show label distribution
    print(f"\n[*] Label distribution:")
    print(f"    beard: {df['beard'].value_counts().to_dict()}")
    print(f"    mustache: {df['mustache'].value_counts().to_dict()}")
    print(f"    glasses_binary: {df['glasses_binary'].value_counts().to_dict()}")
    print(f"    hair_color_label: {df['hair_color_label'].value_counts().to_dict()}")
    print(f"    hair_length: {df['hair_length'].value_counts().to_dict()}")
    
    # Show lot distribution
    if 'filename' in df.columns:
        try:
            # Extract lot from filename (e.g., 's1' from 's1_00000.png')
            lots = df['filename'].str.split('_').str[0].value_counts()
            print(f"\n[*] Images per lot:")
            for lot, count in lots.items():
                print(f"    {lot}: {count} images")
        except Exception as e:
            print(f"\n[!] Could not extract lot distribution: {type(e).__name__}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Map merged annotations to training CSV format"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/annotations/merged_labels.csv',
        help='Input path for merged labels CSV (default: data/annotations/merged_labels.csv)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/annotations/mapped_train.csv',
        help='Output path for mapped training CSV (default: data/annotations/mapped_train.csv)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(" MAP ANNOTATIONS TO TRAINING FORMAT")
    print("=" * 60)
    print(f" Input file: {args.input}")
    print(f" Output file: {args.output}")
    print("=" * 60)
    
    # Load merged labels
    df = load_merged_labels(args.input)
    
    if df.empty:
        print("\n[!] No data loaded. Exiting.")
        return
    
    # Clean filenames
    df = clean_filenames(df)
    
    if df.empty:
        print("\n[!] No valid data after cleaning. Exiting.")
        return
    
    # Validate labels
    df = validate_labels(df)
    
    if df.empty:
        print("\n[!] No valid data after validation. Exiting.")
        return
    
    # Map to training format
    df_mapped = map_to_training_format(df)
    
    if df_mapped.empty:
        print("\n[!] No data after mapping. Exiting.")
        return
    
    # Save result
    save_mapped_labels(df_mapped, args.output)
    
    print("\n" + "=" * 60)
    print(" MAPPING COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
