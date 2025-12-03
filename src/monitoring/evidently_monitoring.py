"""
Evidently Monitoring - Data and Model Drift Detection

This module provides functionality to monitor data drift and model performance
using Evidently AI for the face attribute classification model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

try:
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset, ClassificationPreset
    from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric
except ImportError:
    print("Warning: Evidently not installed. Install with: pip install evidently")


class EvidentlyMonitor:
    """
    Monitor for data drift and model performance using Evidently AI
    
    Attributes monitored:
    - beard (binary)
    - mustache (binary) 
    - glasses (binary)
    - hair_color (5 classes)
    - hair_length (3 classes)
    """
    
    def __init__(self, reference_data_path: Optional[str] = None):
        """
        Initialize the Evidently monitor
        
        Args:
            reference_data_path: Path to reference data CSV. If None, will look for default.
        """
        self.reference_data_path = reference_data_path or "src/monitoring/reference_data/reference.csv"
        self.reports_dir = Path("src/monitoring/reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Define column mapping for face attributes
        self.column_mapping = ColumnMapping(
            target=None,  # No single target for multi-head model
            prediction=None,
            numerical_features=[],
            categorical_features=['beard', 'mustache', 'glasses', 'hair_color', 'hair_length']
        )
        
        self.reference_data = self._load_reference_data()
    
    def _load_reference_data(self) -> Optional[pd.DataFrame]:
        """Load reference data for comparison"""
        ref_path = Path(self.reference_data_path)
        
        if ref_path.exists():
            try:
                df = pd.read_csv(ref_path)
                print(f"✓ Loaded reference data: {len(df)} samples")
                return df
            except Exception as e:
                print(f"⚠️ Error loading reference data: {e}")
                return None
        else:
            print(f"⚠️ Reference data not found at {ref_path}")
            return None
    
    def set_reference_data(self, data: pd.DataFrame):
        """
        Set and save reference data
        
        Args:
            data: DataFrame with reference data
        """
        ref_path = Path(self.reference_data_path)
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        
        data.to_csv(ref_path, index=False)
        self.reference_data = data
        print(f"✓ Reference data saved: {len(data)} samples")
    
    def generate_data_drift_report(self, current_data: pd.DataFrame) -> Report:
        """
        Generate data drift report comparing current data with reference
        
        Args:
            current_data: Current dataset to compare
            
        Returns:
            Evidently Report object
        """
        if self.reference_data is None:
            print("⚠️ No reference data available. Setting current data as reference.")
            self.set_reference_data(current_data)
            return None
        
        # Create report with data drift preset
        report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
        ])
        
        # Add individual column drift metrics for each attribute
        for col in ['beard', 'mustache', 'glasses', 'hair_color', 'hair_length']:
            if col in current_data.columns:
                report.metrics.append(ColumnDriftMetric(column_name=col))
        
        # Run the report
        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        return report
    
    def generate_model_performance_report(
        self, 
        predictions: pd.DataFrame,
        targets: pd.DataFrame
    ) -> Report:
        """
        Generate model performance report
        
        Args:
            predictions: DataFrame with model predictions
            targets: DataFrame with true labels
            
        Returns:
            Evidently Report object
        """
        # Combine predictions and targets
        data = pd.concat([predictions, targets], axis=1)
        
        # Create classification report
        report = Report(metrics=[
            ClassificationPreset(),
        ])
        
        # Run the report
        report.run(
            reference_data=None,
            current_data=data,
            column_mapping=self.column_mapping
        )
        
        return report
    
    def generate_target_drift_report(
        self,
        current_predictions: pd.DataFrame
    ) -> Report:
        """
        Generate target drift report for predictions
        
        Args:
            current_predictions: DataFrame with current model predictions
            
        Returns:
            Evidently Report object
        """
        if self.reference_data is None:
            print("⚠️ No reference data available")
            return None
        
        # Create dataset drift report
        report = Report(metrics=[
            DatasetDriftMetric(),
        ])
        
        # Run the report
        report.run(
            reference_data=self.reference_data,
            current_data=current_predictions,
            column_mapping=self.column_mapping
        )
        
        return report
    
    def check_drift_threshold(
        self, 
        current_data: pd.DataFrame = None,
        threshold: float = 0.1
    ) -> bool:
        """
        Check if drift exceeds threshold
        
        Args:
            current_data: Current dataset to check
            threshold: Drift score threshold (0-1)
            
        Returns:
            True if drift detected, False otherwise
        """
        if current_data is None:
            # Load current data from default location
            data_path = Path("data/annotations/mapped_train.csv")
            if data_path.exists():
                current_data = pd.read_csv(data_path)
            else:
                print("⚠️ No current data available")
                return False
        
        report = self.generate_data_drift_report(current_data)
        
        if report is None:
            return False
        
        # Extract drift metrics
        try:
            # Get the report as dictionary
            report_dict = report.as_dict()
            
            # Check dataset drift score
            metrics = report_dict.get('metrics', [])
            
            for metric in metrics:
                if metric.get('metric') == 'DatasetDriftMetric':
                    drift_score = metric.get('result', {}).get('drift_share', 0)
                    
                    if drift_score > threshold:
                        print(f"⚠️ Drift detected! Score: {drift_score:.4f} > Threshold: {threshold}")
                        return True
            
            print(f"✓ No significant drift detected (threshold: {threshold})")
            return False
            
        except Exception as e:
            print(f"⚠️ Error checking drift: {e}")
            return False
    
    def save_html_report(
        self, 
        report: Report, 
        filename: Optional[str] = None
    ) -> str:
        """
        Save report as HTML
        
        Args:
            report: Evidently Report object
            filename: Output filename (without extension)
            
        Returns:
            Path to saved report
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"drift_report_{timestamp}"
        
        output_path = self.reports_dir / f"{filename}.html"
        report.save_html(str(output_path))
        
        print(f"✓ Report saved to: {output_path}")
        return str(output_path)
    
    def get_drift_summary(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary of drift metrics
        
        Args:
            current_data: Current dataset to analyze
            
        Returns:
            Dictionary with drift summary
        """
        report = self.generate_data_drift_report(current_data)
        
        if report is None:
            return {}
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'n_samples_reference': len(self.reference_data) if self.reference_data is not None else 0,
            'n_samples_current': len(current_data),
            'attributes_checked': ['beard', 'mustache', 'glasses', 'hair_color', 'hair_length'],
            'drift_detected': self.check_drift_threshold(current_data),
        }
        
        return summary


def create_monitor(reference_data_path: Optional[str] = None) -> EvidentlyMonitor:
    """
    Factory function to create an EvidentlyMonitor instance
    
    Args:
        reference_data_path: Path to reference data
        
    Returns:
        EvidentlyMonitor instance
    """
    return EvidentlyMonitor(reference_data_path)
