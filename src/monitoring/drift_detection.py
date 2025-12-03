"""
Drift detection utilities - PSI, KL divergence, feature-level drift
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy import stats
from scipy.special import kl_div


def calculate_psi(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 10
) -> float:
    """
    Calculate Population Stability Index (PSI)
    
    PSI measures the shift in distribution between two datasets
    PSI < 0.1: No significant change
    0.1 <= PSI < 0.2: Moderate change
    PSI >= 0.2: Significant change
    
    Args:
        reference: Reference distribution
        current: Current distribution
        bins: Number of bins for discretization
    
    Returns:
        PSI score
    """
    # Create bins based on reference distribution
    breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)  # Remove duplicates
    
    if len(breakpoints) < 2:
        return 0.0
    
    # Bin both distributions
    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    curr_counts, _ = np.histogram(current, bins=breakpoints)
    
    # Calculate proportions
    ref_props = ref_counts / len(reference)
    curr_props = curr_counts / len(current)
    
    # Avoid division by zero
    ref_props = np.where(ref_props == 0, 0.0001, ref_props)
    curr_props = np.where(curr_props == 0, 0.0001, curr_props)
    
    # Calculate PSI
    psi = np.sum((curr_props - ref_props) * np.log(curr_props / ref_props))
    
    return float(psi)


def calculate_kl_divergence(p: np.ndarray, q: np.ndarray, bins: int = 10) -> float:
    """
    Calculate Kullback-Leibler divergence
    
    Measures how one probability distribution diverges from a reference
    
    Args:
        p: Reference distribution
        q: Current distribution
        bins: Number of bins
    
    Returns:
        KL divergence
    """
    # Create bins
    min_val = min(p.min(), q.min())
    max_val = max(p.max(), q.max())
    bin_edges = np.linspace(min_val, max_val, bins + 1)
    
    # Calculate histograms
    p_hist, _ = np.histogram(p, bins=bin_edges)
    q_hist, _ = np.histogram(q, bins=bin_edges)
    
    # Normalize to probabilities
    p_prob = (p_hist + 1e-10) / (p_hist.sum() + 1e-10 * bins)
    q_prob = (q_hist + 1e-10) / (q_hist.sum() + 1e-10 * bins)
    
    # Calculate KL divergence
    kl = np.sum(kl_div(p_prob, q_prob))
    
    return float(kl)


def detect_feature_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    threshold: float = 0.1
) -> Dict[str, Dict]:
    """
    Detect drift for each feature
    
    Args:
        reference_df: Reference dataset
        current_df: Current dataset
        threshold: PSI threshold for drift detection
    
    Returns:
        Dictionary with drift metrics per feature
    """
    results = {}
    
    for col in reference_df.columns:
        if col not in current_df.columns:
            continue
        
        ref_values = reference_df[col].values
        curr_values = current_df[col].values
        
        # Calculate PSI
        psi = calculate_psi(ref_values, curr_values)
        
        # Calculate KL divergence
        kl = calculate_kl_divergence(ref_values, curr_values)
        
        # Perform statistical test
        if reference_df[col].dtype in ['int64', 'int32', 'object', 'category']:
            # Chi-square test for categorical
            ref_counts = pd.Series(ref_values).value_counts()
            curr_counts = pd.Series(curr_values).value_counts()
            
            # Align indices
            all_values = set(ref_counts.index) | set(curr_counts.index)
            ref_aligned = np.array([ref_counts.get(v, 0) for v in all_values])
            curr_aligned = np.array([curr_counts.get(v, 0) for v in all_values])
            
            chi2, p_value = stats.chisquare(curr_aligned + 1, ref_aligned + 1)
        else:
            # KS test for continuous
            ks_stat, p_value = stats.ks_2samp(ref_values, curr_values)
        
        # Determine if drift detected
        drift_detected = psi > threshold or p_value < 0.05
        
        results[col] = {
            'psi': psi,
            'kl_divergence': kl,
            'p_value': p_value,
            'drift_detected': drift_detected,
            'severity': 'high' if psi > 0.2 else ('medium' if psi > threshold else 'low')
        }
    
    return results


def detect_prediction_drift(
    ref_predictions: Dict[str, np.ndarray],
    curr_predictions: Dict[str, np.ndarray],
    threshold: float = 0.1
) -> Dict[str, Dict]:
    """
    Detect drift in model predictions
    
    Args:
        ref_predictions: Reference predictions (dict of attribute -> predictions)
        curr_predictions: Current predictions
        threshold: PSI threshold
    
    Returns:
        Dictionary with drift metrics per attribute
    """
    results = {}
    
    for attr in ref_predictions.keys():
        if attr not in curr_predictions:
            continue
        
        ref_preds = ref_predictions[attr]
        curr_preds = curr_predictions[attr]
        
        # Calculate PSI
        psi = calculate_psi(ref_preds, curr_preds)
        
        # Calculate distribution statistics
        ref_mean = float(np.mean(ref_preds))
        curr_mean = float(np.mean(curr_preds))
        ref_std = float(np.std(ref_preds))
        curr_std = float(np.std(curr_preds))
        
        # Statistical test
        t_stat, p_value = stats.ttest_ind(ref_preds, curr_preds)
        
        drift_detected = psi > threshold or p_value < 0.05
        
        results[attr] = {
            'psi': psi,
            'p_value': p_value,
            'drift_detected': drift_detected,
            'ref_mean': ref_mean,
            'curr_mean': curr_mean,
            'ref_std': ref_std,
            'curr_std': curr_std,
            'mean_shift': abs(curr_mean - ref_mean),
        }
    
    return results


def generate_drift_summary(
    feature_drift: Dict[str, Dict],
    prediction_drift: Dict[str, Dict] = None
) -> Dict:
    """
    Generate summary of drift analysis
    
    Args:
        feature_drift: Feature-level drift metrics
        prediction_drift: Prediction-level drift metrics (optional)
    
    Returns:
        Summary dictionary
    """
    # Count features with drift
    total_features = len(feature_drift)
    features_with_drift = sum(1 for v in feature_drift.values() if v['drift_detected'])
    drift_share = features_with_drift / total_features if total_features > 0 else 0
    
    # Get high severity drifts
    high_severity = [
        k for k, v in feature_drift.items()
        if v['severity'] == 'high'
    ]
    
    summary = {
        'total_features': total_features,
        'features_with_drift': features_with_drift,
        'drift_share': drift_share,
        'high_severity_features': high_severity,
        'avg_psi': np.mean([v['psi'] for v in feature_drift.values()]),
        'max_psi': max([v['psi'] for v in feature_drift.values()]) if feature_drift else 0,
    }
    
    if prediction_drift:
        summary['prediction_drift'] = {
            'attributes_with_drift': sum(1 for v in prediction_drift.values() if v['drift_detected']),
            'total_attributes': len(prediction_drift),
        }
    
    return summary


if __name__ == '__main__':
    # Example usage
    print("Drift Detection - Example Usage")
    
    # Simulate data
    np.random.seed(42)
    reference = np.random.normal(0, 1, 1000)
    current_no_drift = np.random.normal(0, 1, 1000)
    current_with_drift = np.random.normal(0.5, 1.2, 1000)
    
    # Calculate PSI
    psi_no_drift = calculate_psi(reference, current_no_drift)
    psi_with_drift = calculate_psi(reference, current_with_drift)
    
    print(f"\nPSI (no drift): {psi_no_drift:.4f}")
    print(f"PSI (with drift): {psi_with_drift:.4f}")
    
    # Test with dataframe
    ref_df = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.choice([0, 1], 1000),
    })
    
    curr_df = pd.DataFrame({
        'feature1': np.random.normal(0.3, 1.1, 1000),
        'feature2': np.random.choice([0, 1], 1000, p=[0.4, 0.6]),
    })
    
    # Detect drift
    drift_results = detect_feature_drift(ref_df, curr_df)
    
    print("\nFeature drift analysis:")
    for feature, metrics in drift_results.items():
        print(f"\n{feature}:")
        print(f"  PSI: {metrics['psi']:.4f}")
        print(f"  Drift detected: {metrics['drift_detected']}")
        print(f"  Severity: {metrics['severity']}")
    
    # Summary
    summary = generate_drift_summary(drift_results)
    print(f"\nSummary:")
    print(f"  Drift share: {summary['drift_share']:.2%}")
    print(f"  Average PSI: {summary['avg_psi']:.4f}")
