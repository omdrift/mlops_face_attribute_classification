#!/usr/bin/env python3
"""
Run Drift Check Script

This script checks for data and model drift using Evidently AI.
Can be used in GitHub Actions and Airflow.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from monitoring.evidently_monitoring import EvidentlyMonitor
from monitoring.data_drift import detect_distribution_drift, create_drift_alert
from monitoring.model_drift import check_model_drift, create_performance_alert


def run_data_drift_check(threshold: float = 0.1):
    """
    Run data drift detection
    
    Args:
        threshold: Drift threshold (0-1)
        
    Returns:
        Exit code: 0 = no drift, 1 = drift detected
    """
    print("=" * 80)
    print("Data Drift Check")
    print("=" * 80)
    
    try:
        # Check distribution drift
        drift_results = detect_distribution_drift()
        
        if 'error' in drift_results:
            print(f"❌ Error: {drift_results['error']}")
            if 'message' in drift_results:
                print(f"   {drift_results['message']}")
            return 0  # Don't fail on first run
        
        # Print alert
        alert_message = create_drift_alert(drift_results)
        print(alert_message)
        
        # Check if drift exceeds threshold
        drift_summary = drift_results.get('drift_summary', {})
        drift_percentage = drift_summary.get('drift_percentage', 0)
        
        print(f"\nDrift Summary:")
        print(f"  - Attributes checked: {drift_summary.get('total_attributes', 0)}")
        print(f"  - Attributes with drift: {drift_summary.get('attributes_with_drift', 0)}")
        print(f"  - Drift percentage: {drift_percentage:.2%}")
        print(f"  - Threshold: {threshold:.2%}")
        
        if drift_percentage > threshold:
            print(f"\n⚠️ DRIFT DETECTED: {drift_percentage:.2%} > {threshold:.2%}")
            return 1
        else:
            print(f"\n✓ No significant drift: {drift_percentage:.2%} <= {threshold:.2%}")
            return 0
            
    except Exception as e:
        print(f"❌ Error during drift check: {e}")
        import traceback
        traceback.print_exc()
        return 1


def run_model_drift_check(threshold: float = 0.05):
    """
    Run model performance drift detection
    
    Args:
        threshold: Performance drop threshold (e.g., 0.05 = 5%)
        
    Returns:
        Exit code: 0 = no drift, 1 = drift detected
    """
    print("\n" + "=" * 80)
    print("Model Performance Drift Check")
    print("=" * 80)
    
    try:
        drift_results = check_model_drift(threshold=threshold)
        
        if 'error' in drift_results:
            print(f"❌ Error: {drift_results['error']}")
            return 0  # Don't fail if no history yet
        
        # Print alert
        alert_message = create_performance_alert(drift_results)
        print(alert_message)
        
        if drift_results.get('drift_detected', False):
            print(f"\n⚠️ MODEL DRIFT DETECTED")
            return 1
        else:
            print(f"\n✓ No significant model performance drift")
            return 0
            
    except Exception as e:
        print(f"❌ Error during model drift check: {e}")
        import traceback
        traceback.print_exc()
        return 0  # Don't fail on error


def run_evidently_check(threshold: float = 0.1):
    """
    Run Evidently monitoring check
    
    Args:
        threshold: Drift threshold
        
    Returns:
        Exit code: 0 = no drift, 1 = drift detected
    """
    print("\n" + "=" * 80)
    print("Evidently AI Drift Check")
    print("=" * 80)
    
    try:
        import pandas as pd
        
        monitor = EvidentlyMonitor()
        
        # Load current data
        data_path = Path("data/annotations/mapped_train.csv")
        if not data_path.exists():
            print(f"⚠️ Data file not found: {data_path}")
            return 0
        
        current_data = pd.read_csv(data_path)
        print(f"Loaded {len(current_data)} samples from {data_path}")
        
        # Check drift
        drift_detected = monitor.check_drift_threshold(current_data, threshold=threshold)
        
        # Generate report
        report = monitor.generate_data_drift_report(current_data)
        
        if report:
            # Save HTML report
            report_path = monitor.save_html_report(report)
            print(f"\n📊 HTML report saved to: {report_path}")
        
        # Get drift summary
        summary = monitor.get_drift_summary(current_data)
        print(f"\n📈 Drift Summary:")
        print(f"  - Timestamp: {summary.get('timestamp')}")
        print(f"  - Reference samples: {summary.get('n_samples_reference')}")
        print(f"  - Current samples: {summary.get('n_samples_current')}")
        print(f"  - Drift detected: {summary.get('drift_detected')}")
        
        if drift_detected:
            print(f"\n⚠️ DRIFT DETECTED (threshold: {threshold})")
            return 1
        else:
            print(f"\n✓ No drift detected (threshold: {threshold})")
            return 0
            
    except ImportError:
        print("⚠️ Evidently not installed. Install with: pip install evidently")
        return 0
    except Exception as e:
        print(f"❌ Error during Evidently check: {e}")
        import traceback
        traceback.print_exc()
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Run drift checks for ML model monitoring"
    )
    parser.add_argument(
        '--data-threshold',
        type=float,
        default=0.1,
        help='Data drift threshold (default: 0.1)'
    )
    parser.add_argument(
        '--model-threshold',
        type=float,
        default=0.05,
        help='Model performance drift threshold (default: 0.05)'
    )
    parser.add_argument(
        '--check',
        choices=['data', 'model', 'evidently', 'all'],
        default='all',
        help='Type of drift check to run (default: all)'
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("MLOps Drift Check")
    print(f"{'='*80}\n")
    
    exit_codes = []
    
    # Run requested checks
    if args.check in ['data', 'all']:
        exit_codes.append(run_data_drift_check(args.data_threshold))
    
    if args.check in ['model', 'all']:
        exit_codes.append(run_model_drift_check(args.model_threshold))
    
    if args.check in ['evidently', 'all']:
        exit_codes.append(run_evidently_check(args.data_threshold))
    
    # Return 1 if any check detected drift
    final_exit_code = max(exit_codes) if exit_codes else 0
    
    print(f"\n{'='*80}")
    if final_exit_code == 0:
        print("✓ All drift checks passed")
    else:
        print("⚠️ Drift detected - review reports and consider model retraining")
    print(f"{'='*80}\n")
    
    return final_exit_code


if __name__ == "__main__":
    sys.exit(main())
