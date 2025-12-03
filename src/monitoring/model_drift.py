"""
Model Drift Detection

This module detects drift in model performance over time.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


class ModelPerformanceTracker:
    """
    Track model performance metrics over time and detect drift
    """
    
    def __init__(self, metrics_dir: str = "metrics/history"):
        """
        Initialize the performance tracker
        
        Args:
            metrics_dir: Directory to store metrics history
        """
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.metrics_dir / "performance_history.json"
        
        # Load existing history
        self.history = self._load_history()
    
    def _load_history(self) -> List[Dict[str, Any]]:
        """Load performance history"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Error loading history: {e}")
                return []
        return []
    
    def _save_history(self):
        """Save performance history"""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def add_metrics(
        self,
        accuracy: float,
        precision: Dict[str, float],
        recall: Dict[str, float],
        f1: Dict[str, float],
        model_version: Optional[str] = None
    ):
        """
        Add new metrics to history
        
        Args:
            accuracy: Overall accuracy
            precision: Precision per attribute
            recall: Recall per attribute
            f1: F1 score per attribute
            model_version: Optional model version identifier
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'model_version': model_version or 'unknown',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        self.history.append(entry)
        self._save_history()
        
        print(f"✓ Metrics added to history (total entries: {len(self.history)})")
    
    def get_recent_metrics(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get the n most recent metric entries
        
        Args:
            n: Number of recent entries to return
            
        Returns:
            List of recent metric entries
        """
        return self.history[-n:] if len(self.history) >= n else self.history
    
    def detect_performance_drift(
        self,
        threshold: float = 0.05,
        window_size: int = 5
    ) -> Dict[str, Any]:
        """
        Detect drift in model performance
        
        Args:
            threshold: Performance drop threshold (e.g., 0.05 = 5% drop)
            window_size: Number of recent evaluations to compare
            
        Returns:
            Dictionary with drift detection results
        """
        if len(self.history) < window_size + 1:
            return {
                'drift_detected': False,
                'message': f'Insufficient history (need at least {window_size + 1} entries)'
            }
        
        # Get baseline (first metrics) and recent metrics
        baseline = self.history[0]
        recent = self.history[-window_size:]
        
        # Calculate average recent performance
        recent_accuracy = np.mean([m['accuracy'] for m in recent])
        baseline_accuracy = baseline['accuracy']
        
        # Calculate drop
        accuracy_drop = baseline_accuracy - recent_accuracy
        
        result = {
            'drift_detected': accuracy_drop > threshold,
            'baseline_accuracy': baseline_accuracy,
            'recent_accuracy': recent_accuracy,
            'accuracy_drop': accuracy_drop,
            'threshold': threshold,
            'window_size': window_size
        }
        
        # Check per-attribute performance
        attribute_drift = {}
        
        for attr in ['beard', 'mustache', 'glasses', 'hair_color', 'hair_length']:
            if attr in baseline.get('f1', {}):
                baseline_f1 = baseline['f1'][attr]
                recent_f1 = np.mean([m['f1'].get(attr, 0) for m in recent if attr in m.get('f1', {})])
                
                f1_drop = baseline_f1 - recent_f1
                
                attribute_drift[attr] = {
                    'baseline_f1': baseline_f1,
                    'recent_f1': recent_f1,
                    'f1_drop': f1_drop,
                    'drift_detected': f1_drop > threshold
                }
        
        result['attribute_drift'] = attribute_drift
        
        return result
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get summary of model performance over time
        
        Returns:
            Dictionary with performance summary
        """
        if not self.history:
            return {'error': 'No performance history available'}
        
        # Extract accuracy values
        accuracies = [m['accuracy'] for m in self.history]
        
        summary = {
            'n_evaluations': len(self.history),
            'first_evaluation': self.history[0]['timestamp'],
            'last_evaluation': self.history[-1]['timestamp'],
            'accuracy_stats': {
                'mean': np.mean(accuracies),
                'std': np.std(accuracies),
                'min': np.min(accuracies),
                'max': np.max(accuracies),
                'current': accuracies[-1]
            }
        }
        
        # Trend analysis (simple linear regression)
        if len(accuracies) >= 3:
            x = np.arange(len(accuracies))
            coeffs = np.polyfit(x, accuracies, 1)
            trend_slope = coeffs[0]
            
            summary['trend'] = {
                'slope': trend_slope,
                'direction': 'improving' if trend_slope > 0 else 'degrading' if trend_slope < 0 else 'stable'
            }
        
        return summary


def check_model_drift(
    current_metrics_path: str = "metrics/train_metrics.json",
    threshold: float = 0.05
) -> Dict[str, Any]:
    """
    Check if model performance has drifted
    
    Args:
        current_metrics_path: Path to current metrics file
        threshold: Performance drop threshold
        
    Returns:
        Dictionary with drift check results
    """
    tracker = ModelPerformanceTracker()
    
    # Load current metrics
    metrics_path = Path(current_metrics_path)
    
    if not metrics_path.exists():
        return {'error': 'Current metrics file not found'}
    
    with open(metrics_path, 'r') as f:
        current_metrics = json.load(f)
    
    # Add current metrics to history (if not already added)
    # This is a simplified version - in practice, you'd check timestamps
    
    # Detect drift
    drift_results = tracker.detect_performance_drift(threshold=threshold)
    
    return drift_results


def create_performance_alert(drift_results: Dict[str, Any]) -> str:
    """
    Create a human-readable performance alert
    
    Args:
        drift_results: Results from detect_performance_drift
        
    Returns:
        Alert message string
    """
    if 'error' in drift_results:
        return f"Error: {drift_results['error']}"
    
    if not drift_results['drift_detected']:
        return "✓ No significant model performance drift detected"
    
    accuracy_drop = drift_results['accuracy_drop']
    threshold = drift_results['threshold']
    
    message = f"⚠️ Model performance drift detected!\n"
    message += f"  Accuracy dropped by {accuracy_drop:.2%} (threshold: {threshold:.2%})\n"
    message += f"  Baseline: {drift_results['baseline_accuracy']:.4f}\n"
    message += f"  Recent: {drift_results['recent_accuracy']:.4f}\n"
    
    # Check attribute-level drift
    attr_drift_count = sum(
        1 for attr_info in drift_results['attribute_drift'].values()
        if attr_info['drift_detected']
    )
    
    if attr_drift_count > 0:
        message += f"\n  Attributes with drift: {attr_drift_count}\n"
        
        for attr, info in drift_results['attribute_drift'].items():
            if info['drift_detected']:
                message += f"    - {attr}: F1 dropped by {info['f1_drop']:.4f}\n"
    
    return message


if __name__ == "__main__":
    # Example usage
    tracker = ModelPerformanceTracker()
    
    print("Performance summary:")
    summary = tracker.get_performance_summary()
    print(json.dumps(summary, indent=2))
    
    print("\nChecking for drift...")
    drift_results = check_model_drift()
    print(create_performance_alert(drift_results))
