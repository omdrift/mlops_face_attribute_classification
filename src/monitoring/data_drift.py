"""
Data Drift Detection

This module detects drift in the face attribute data distributions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from scipy import stats


def load_data(data_path: str = "data/annotations/mapped_train.csv") -> pd.DataFrame:
    """Load face attribute data"""
    path = Path(data_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    
    df = pd.read_csv(path)
    return df


def check_data_quality() -> Dict[str, Any]:
    """
    Check data quality metrics
    
    Returns:
        Dictionary with quality metrics
    """
    df = load_data()
    
    quality_report = {
        'n_samples': len(df),
        'n_features': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
    }
    
    # Check value ranges for binary attributes
    for attr in ['beard', 'mustache', 'glasses']:
        if attr in df.columns:
            unique_vals = sorted(df[attr].unique())
            quality_report[f'{attr}_values'] = unique_vals
    
    # Check hair attributes
    if 'hair_color' in df.columns:
        quality_report['hair_color_distribution'] = df['hair_color'].value_counts().to_dict()
    
    if 'hair_length' in df.columns:
        quality_report['hair_length_distribution'] = df['hair_length'].value_counts().to_dict()
    
    return quality_report


def compare_distributions(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    attribute: str
) -> Dict[str, Any]:
    """
    Compare distributions for a specific attribute
    
    Args:
        reference_data: Reference dataset
        current_data: Current dataset
        attribute: Attribute name to compare
        
    Returns:
        Dictionary with comparison metrics
    """
    if attribute not in reference_data.columns or attribute not in current_data.columns:
        return {'error': f'Attribute {attribute} not found'}
    
    ref_values = reference_data[attribute].dropna()
    curr_values = current_data[attribute].dropna()
    
    # Calculate distribution statistics
    result = {
        'attribute': attribute,
        'reference_mean': float(ref_values.mean()),
        'current_mean': float(curr_values.mean()),
        'reference_std': float(ref_values.std()),
        'current_std': float(curr_values.std()),
    }
    
    # Chi-square test for categorical data
    try:
        # Get value counts for both datasets
        ref_counts = ref_values.value_counts(normalize=True)
        curr_counts = curr_values.value_counts(normalize=True)
        
        # Align indices
        all_values = sorted(set(ref_counts.index) | set(curr_counts.index))
        ref_freq = [ref_counts.get(v, 0) for v in all_values]
        curr_freq = [curr_counts.get(v, 0) for v in all_values]
        
        # Chi-square test
        chi2, p_value = stats.chisquare(curr_freq, ref_freq)
        
        result['chi2_statistic'] = float(chi2)
        result['p_value'] = float(p_value)
        result['drift_detected'] = p_value < 0.05
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


def detect_distribution_drift(
    reference_path: str = "src/monitoring/reference_data/reference.csv",
    current_path: str = "data/annotations/mapped_train.csv",
    attributes: Optional[list] = None
) -> Dict[str, Any]:
    """
    Detect drift in attribute distributions
    
    Args:
        reference_path: Path to reference data
        current_path: Path to current data
        attributes: List of attributes to check (default: all)
        
    Returns:
        Dictionary with drift detection results
    """
    # Load data
    ref_path = Path(reference_path)
    curr_path = Path(current_path)
    
    if not ref_path.exists():
        return {
            'error': 'Reference data not found',
            'message': 'Run initial training to create reference data'
        }
    
    if not curr_path.exists():
        return {'error': 'Current data not found'}
    
    reference_data = pd.read_csv(ref_path)
    current_data = pd.read_csv(curr_path)
    
    # Default attributes
    if attributes is None:
        attributes = ['beard', 'mustache', 'glasses', 'hair_color', 'hair_length']
    
    # Compare each attribute
    results = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'n_reference': len(reference_data),
        'n_current': len(current_data),
        'attributes': {}
    }
    
    drift_count = 0
    
    for attr in attributes:
        if attr in reference_data.columns and attr in current_data.columns:
            comparison = compare_distributions(reference_data, current_data, attr)
            results['attributes'][attr] = comparison
            
            if comparison.get('drift_detected', False):
                drift_count += 1
    
    results['drift_summary'] = {
        'total_attributes': len(attributes),
        'attributes_with_drift': drift_count,
        'drift_percentage': drift_count / len(attributes) if attributes else 0
    }
    
    return results


def create_drift_alert(drift_results: Dict[str, Any]) -> str:
    """
    Create a human-readable drift alert message
    
    Args:
        drift_results: Results from detect_distribution_drift
        
    Returns:
        Alert message string
    """
    if 'error' in drift_results:
        return f"Error: {drift_results['error']}"
    
    summary = drift_results['drift_summary']
    
    if summary['attributes_with_drift'] == 0:
        return "✓ No drift detected in any attributes"
    
    message = f"⚠️ Drift detected in {summary['attributes_with_drift']}/{summary['total_attributes']} attributes:\n"
    
    for attr, results in drift_results['attributes'].items():
        if results.get('drift_detected', False):
            p_value = results.get('p_value', 0)
            message += f"  - {attr}: p-value={p_value:.4f}\n"
    
    return message


if __name__ == "__main__":
    # Example usage
    print("Checking data quality...")
    quality = check_data_quality()
    print(f"Data quality report: {quality}")
    
    print("\nDetecting drift...")
    drift_results = detect_distribution_drift()
    
    if 'error' not in drift_results:
        print(create_drift_alert(drift_results))
