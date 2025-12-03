"""
Tests for monitoring components
"""
import pytest
import numpy as np
import pandas as pd
from src.monitoring.drift_detection import (
    calculate_psi,
    calculate_kl_divergence,
    detect_feature_drift,
    detect_prediction_drift,
    generate_drift_summary
)
from src.monitoring.report_generator import (
    generate_html_report,
    generate_json_summary
)


class TestDriftDetection:
    """Tests for drift detection functions"""
    
    def test_calculate_psi_no_drift(self):
        """Test PSI calculation with no drift"""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 1000)
        current = np.random.normal(0, 1, 1000)
        
        psi = calculate_psi(reference, current)
        
        assert isinstance(psi, float)
        assert psi >= 0
        assert psi < 0.1  # Should be low for similar distributions
    
    def test_calculate_psi_with_drift(self):
        """Test PSI calculation with drift"""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 1000)
        current = np.random.normal(1, 1.5, 1000)  # Different distribution
        
        psi = calculate_psi(reference, current)
        
        assert isinstance(psi, float)
        assert psi > 0.1  # Should be high for different distributions
    
    def test_calculate_kl_divergence(self):
        """Test KL divergence calculation"""
        np.random.seed(42)
        p = np.random.normal(0, 1, 1000)
        q = np.random.normal(0.5, 1.2, 1000)
        
        kl = calculate_kl_divergence(p, q)
        
        assert isinstance(kl, float)
        assert kl >= 0
    
    def test_detect_feature_drift(self):
        """Test feature-level drift detection"""
        np.random.seed(42)
        
        # Create sample dataframes
        ref_df = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.choice([0, 1], 1000),
        })
        
        curr_df = pd.DataFrame({
            'feature1': np.random.normal(0.3, 1.1, 1000),
            'feature2': np.random.choice([0, 1], 1000, p=[0.4, 0.6]),
        })
        
        results = detect_feature_drift(ref_df, curr_df, threshold=0.1)
        
        assert isinstance(results, dict)
        assert 'feature1' in results
        assert 'feature2' in results
        assert 'psi' in results['feature1']
        assert 'drift_detected' in results['feature1']
    
    def test_detect_prediction_drift(self):
        """Test prediction drift detection"""
        np.random.seed(42)
        
        ref_preds = {
            'beard': np.random.choice([0, 1], 1000),
            'mustache': np.random.choice([0, 1], 1000),
        }
        
        curr_preds = {
            'beard': np.random.choice([0, 1], 1000, p=[0.6, 0.4]),
            'mustache': np.random.choice([0, 1], 1000, p=[0.8, 0.2]),
        }
        
        results = detect_prediction_drift(ref_preds, curr_preds)
        
        assert isinstance(results, dict)
        assert 'beard' in results
        assert 'mustache' in results
        assert 'psi' in results['beard']
        assert 'drift_detected' in results['beard']
    
    def test_generate_drift_summary(self):
        """Test drift summary generation"""
        feature_drift = {
            'feature1': {'psi': 0.15, 'drift_detected': True, 'severity': 'medium'},
            'feature2': {'psi': 0.05, 'drift_detected': False, 'severity': 'low'},
            'feature3': {'psi': 0.25, 'drift_detected': True, 'severity': 'high'},
        }
        
        summary = generate_drift_summary(feature_drift)
        
        assert isinstance(summary, dict)
        assert 'total_features' in summary
        assert 'features_with_drift' in summary
        assert 'drift_share' in summary
        assert summary['total_features'] == 3
        assert summary['features_with_drift'] == 2
        assert summary['drift_share'] == 2/3


class TestReportGeneration:
    """Tests for report generation"""
    
    def test_generate_html_report(self, tmp_path):
        """Test HTML report generation"""
        drift_results = {
            'summary': {
                'total_features': 3,
                'features_with_drift': 1,
                'drift_share': 0.33,
                'avg_psi': 0.12,
            },
            'features': {
                'feature1': {
                    'psi': 0.15,
                    'p_value': 0.03,
                    'drift_detected': True,
                    'severity': 'medium'
                }
            }
        }
        
        output_path = tmp_path / "test_report.html"
        result = generate_html_report(drift_results, str(output_path))
        
        assert output_path.exists()
        assert result == str(output_path)
        
        # Check content
        content = output_path.read_text()
        assert 'Drift Detection Report' in content
        assert 'feature1' in content
    
    def test_generate_json_summary(self, tmp_path):
        """Test JSON summary generation"""
        drift_results = {
            'summary': {'drift_share': 0.3},
            'features': {'feature1': {'psi': 0.15}}
        }
        
        output_path = tmp_path / "test_summary.json"
        result = generate_json_summary(drift_results, str(output_path))
        
        assert output_path.exists()
        assert result == str(output_path)
        
        # Check JSON is valid
        import json
        with open(output_path) as f:
            data = json.load(f)
        
        assert 'timestamp' in data
        assert 'results' in data


class TestPrometheusMetrics:
    """Tests for Prometheus metrics"""
    
    def test_metrics_import(self):
        """Test that Prometheus metrics can be imported"""
        from src.monitoring.prometheus_metrics import (
            api_requests_total,
            model_predictions_total,
            drift_alerts_total
        )
        
        assert api_requests_total is not None
        assert model_predictions_total is not None
        assert drift_alerts_total is not None
    
    def test_record_prediction(self):
        """Test recording predictions"""
        from src.monitoring.prometheus_metrics import record_prediction
        
        # Should not raise exception
        record_prediction('beard', 1)
        record_prediction('mustache', 0)
    
    def test_record_drift_alert(self):
        """Test recording drift alerts"""
        from src.monitoring.prometheus_metrics import record_drift_alert
        
        # Should not raise exception
        record_drift_alert('beard', 'high')
        record_drift_alert('mustache', 'low')
