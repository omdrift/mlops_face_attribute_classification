"""
Evidently AI monitoring class for drift detection and model performance
"""
import os
from datetime import datetime
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np

try:
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
    from evidently.metrics import DatasetDriftMetric, DatasetMissingValuesMetric
except ImportError:
    print("Warning: evidently not installed. Install with: pip install evidently")


class EvidentlyMonitor:
    """
    Main monitoring class using Evidently AI for drift detection
    """
    
    def __init__(self, reference_data: Optional[pd.DataFrame] = None):
        """
        Initialize Evidently monitor
        
        Args:
            reference_data: Reference dataset for comparison (training data)
        """
        self.reference_data = reference_data
        self.column_mapping = ColumnMapping(
            target=None,  # Multi-output model
            prediction=None,
            numerical_features=[],
            categorical_features=['beard', 'mustache', 'glasses', 'hair_color', 'hair_length']
        )
    
    def set_reference_data(self, data: pd.DataFrame):
        """Set reference data for drift detection"""
        self.reference_data = data
        print(f"✓ Reference data set: {len(data)} samples")
    
    def generate_data_drift_report(
        self,
        current_data: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> Report:
        """
        Generate data drift report comparing current data to reference
        
        Args:
            current_data: Current production data
            output_path: Path to save HTML report (optional)
        
        Returns:
            Evidently Report object
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set. Call set_reference_data() first.")
        
        # Create report
        report = Report(metrics=[
            DataDriftPreset(),
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
        ])
        
        # Run report
        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        # Save report
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"reports/evidently/data_drift_report_{timestamp}.html"
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        report.save_html(output_path)
        print(f"✓ Data drift report saved to {output_path}")
        
        return report
    
    def generate_model_performance_report(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        reference_predictions: Dict[str, np.ndarray],
        current_predictions: Dict[str, np.ndarray],
        output_path: Optional[str] = None
    ) -> Report:
        """
        Generate model performance report
        
        Args:
            reference_data: Reference dataset with true labels
            current_data: Current dataset with true labels
            reference_predictions: Model predictions on reference data
            current_predictions: Model predictions on current data
            output_path: Path to save HTML report
        
        Returns:
            Evidently Report object
        """
        # Prepare data with predictions
        ref_df = reference_data.copy()
        curr_df = current_data.copy()
        
        # Add predictions for each attribute
        for attr, preds in reference_predictions.items():
            ref_df[f'{attr}_pred'] = preds
        
        for attr, preds in current_predictions.items():
            curr_df[f'{attr}_pred'] = preds
        
        # Create report with classification metrics
        from evidently.metrics import ClassificationQualityMetric
        
        report = Report(metrics=[
            ClassificationQualityMetric()
        ])
        
        # Run report
        report.run(reference_data=ref_df, current_data=curr_df)
        
        # Save report
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"reports/evidently/performance_report_{timestamp}.html"
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        report.save_html(output_path)
        print(f"✓ Performance report saved to {output_path}")
        
        return report
    
    def generate_target_drift_report(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> Report:
        """
        Generate target drift report
        
        Args:
            reference_data: Reference dataset
            current_data: Current dataset
            output_path: Path to save HTML report
        
        Returns:
            Evidently Report object
        """
        # Create report
        report = Report(metrics=[
            TargetDriftPreset(),
        ])
        
        # Run report for each target attribute
        report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        # Save report
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"reports/evidently/target_drift_report_{timestamp}.html"
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        report.save_html(output_path)
        print(f"✓ Target drift report saved to {output_path}")
        
        return report
    
    def check_drift_threshold(
        self,
        current_data: pd.DataFrame,
        threshold: float = 0.1
    ) -> Tuple[bool, Dict]:
        """
        Check if drift exceeds threshold
        
        Args:
            current_data: Current production data
            threshold: Maximum acceptable drift share (0-1)
        
        Returns:
            Tuple of (drift_detected, drift_metrics)
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set.")
        
        # Create report
        report = Report(metrics=[DatasetDriftMetric()])
        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        # Extract drift metrics
        result = report.as_dict()
        drift_share = result['metrics'][0]['result'].get('drift_share', 0)
        
        drift_detected = drift_share > threshold
        
        drift_metrics = {
            'drift_share': drift_share,
            'threshold': threshold,
            'drift_detected': drift_detected,
            'timestamp': datetime.now().isoformat()
        }
        
        if drift_detected:
            print(f"⚠ Drift detected! Share: {drift_share:.2%} > {threshold:.2%}")
        else:
            print(f"✓ No significant drift. Share: {drift_share:.2%}")
        
        return drift_detected, drift_metrics


def load_reference_data(path: str = 'data/processed/train_data_s1.pt') -> pd.DataFrame:
    """
    Load reference data from training dataset
    
    Args:
        path: Path to training data
    
    Returns:
        DataFrame with reference data
    """
    import torch
    
    if not os.path.exists(path):
        print(f"Warning: Reference data not found at {path}")
        return None
    
    data = torch.load(path)
    X = data['X'].numpy()
    y = data['y'].numpy()
    
    # Create DataFrame
    df = pd.DataFrame(y, columns=['beard', 'mustache', 'glasses', 'hair_color', 'hair_length'])
    
    print(f"✓ Loaded reference data: {len(df)} samples")
    return df


if __name__ == '__main__':
    # Example usage
    print("Evidently Monitor - Example Usage")
    
    # Load reference data
    ref_data = load_reference_data()
    
    if ref_data is not None:
        # Initialize monitor
        monitor = EvidentlyMonitor(reference_data=ref_data)
        
        # Simulate current data (in production, this would be real data)
        current_data = ref_data.sample(n=min(1000, len(ref_data)))
        
        # Generate drift report
        monitor.generate_data_drift_report(current_data)
        
        # Check drift threshold
        drift_detected, metrics = monitor.check_drift_threshold(current_data)
        print(f"\nDrift metrics: {metrics}")
