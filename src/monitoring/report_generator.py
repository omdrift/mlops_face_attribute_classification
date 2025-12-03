"""
Report generator for drift and performance monitoring
"""
import os
import json
from datetime import datetime
from typing import Dict, Optional
import pandas as pd


def generate_html_report(
    drift_results: Dict,
    output_path: str = None,
    title: str = "Drift Detection Report"
) -> str:
    """
    Generate HTML report from drift results
    
    Args:
        drift_results: Dictionary with drift metrics
        output_path: Path to save HTML report
        title: Report title
    
    Returns:
        Path to saved report
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"reports/evidently/drift_report_{timestamp}.html"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Build HTML
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            margin: 15px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-title {{
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
        }}
        .drift-detected {{
            color: #e74c3c;
        }}
        .no-drift {{
            color: #27ae60;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #667eea;
            color: white;
        }}
        .high {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .medium {{
            color: #f39c12;
        }}
        .low {{
            color: #27ae60;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>
"""
    
    # Summary metrics
    if 'summary' in drift_results:
        summary = drift_results['summary']
        html += f"""
    <div class="metric-card">
        <div class="metric-title">Drift Summary</div>
        <table>
            <tr>
                <td>Total Features</td>
                <td><strong>{summary.get('total_features', 0)}</strong></td>
            </tr>
            <tr>
                <td>Features with Drift</td>
                <td><strong class="drift-detected">{summary.get('features_with_drift', 0)}</strong></td>
            </tr>
            <tr>
                <td>Drift Share</td>
                <td><strong>{summary.get('drift_share', 0):.2%}</strong></td>
            </tr>
            <tr>
                <td>Average PSI</td>
                <td><strong>{summary.get('avg_psi', 0):.4f}</strong></td>
            </tr>
        </table>
    </div>
"""
    
    # Feature-level drift
    if 'features' in drift_results:
        html += """
    <div class="metric-card">
        <div class="metric-title">Feature-Level Drift</div>
        <table>
            <thead>
                <tr>
                    <th>Feature</th>
                    <th>PSI</th>
                    <th>P-Value</th>
                    <th>Drift Detected</th>
                    <th>Severity</th>
                </tr>
            </thead>
            <tbody>
"""
        for feature, metrics in drift_results['features'].items():
            drift_status = "✓ Yes" if metrics['drift_detected'] else "✗ No"
            drift_class = "drift-detected" if metrics['drift_detected'] else "no-drift"
            severity_class = metrics.get('severity', 'low')
            
            html += f"""
                <tr>
                    <td><strong>{feature}</strong></td>
                    <td>{metrics['psi']:.4f}</td>
                    <td>{metrics.get('p_value', 0):.4f}</td>
                    <td class="{drift_class}">{drift_status}</td>
                    <td class="{severity_class}">{metrics.get('severity', 'low').upper()}</td>
                </tr>
"""
        html += """
            </tbody>
        </table>
    </div>
"""
    
    # Close HTML
    html += """
</body>
</html>
"""
    
    # Save report
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"✓ HTML report saved to {output_path}")
    return output_path


def generate_json_summary(
    drift_results: Dict,
    output_path: str = None
) -> str:
    """
    Generate JSON summary of drift results
    
    Args:
        drift_results: Drift metrics dictionary
        output_path: Path to save JSON
    
    Returns:
        Path to saved JSON
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"reports/evidently/drift_summary_{timestamp}.json"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Add metadata
    summary = {
        'timestamp': datetime.now().isoformat(),
        'report_type': 'drift_detection',
        'results': drift_results
    }
    
    # Save JSON
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ JSON summary saved to {output_path}")
    return output_path


def log_to_mlflow(
    drift_results: Dict,
    run_id: Optional[str] = None
) -> None:
    """
    Log drift metrics to MLflow
    
    Args:
        drift_results: Drift metrics dictionary
        run_id: MLflow run ID (optional)
    """
    try:
        import mlflow
    except ImportError:
        print("Warning: MLflow not installed")
        return
    
    # Start or resume run
    if run_id:
        mlflow.start_run(run_id=run_id)
    else:
        mlflow.start_run()
    
    try:
        # Log summary metrics
        if 'summary' in drift_results:
            summary = drift_results['summary']
            mlflow.log_metric('drift_share', summary.get('drift_share', 0))
            mlflow.log_metric('avg_psi', summary.get('avg_psi', 0))
            mlflow.log_metric('max_psi', summary.get('max_psi', 0))
            mlflow.log_metric('features_with_drift', summary.get('features_with_drift', 0))
        
        # Log feature-level metrics
        if 'features' in drift_results:
            for feature, metrics in drift_results['features'].items():
                mlflow.log_metric(f'psi_{feature}', metrics['psi'])
                mlflow.log_metric(f'drift_{feature}', 1 if metrics['drift_detected'] else 0)
        
        # Log as artifact
        temp_file = f'/tmp/drift_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(temp_file, 'w') as f:
            json.dump(drift_results, f, indent=2)
        mlflow.log_artifact(temp_file, artifact_path='monitoring')
        os.remove(temp_file)
        
        print("✓ Drift metrics logged to MLflow")
    
    finally:
        mlflow.end_run()


if __name__ == '__main__':
    # Example usage
    print("Report Generator - Example Usage")
    
    # Sample drift results
    drift_results = {
        'summary': {
            'total_features': 5,
            'features_with_drift': 2,
            'drift_share': 0.4,
            'avg_psi': 0.15,
            'max_psi': 0.25,
        },
        'features': {
            'beard': {
                'psi': 0.08,
                'p_value': 0.12,
                'drift_detected': False,
                'severity': 'low'
            },
            'mustache': {
                'psi': 0.25,
                'p_value': 0.001,
                'drift_detected': True,
                'severity': 'high'
            },
            'glasses': {
                'psi': 0.15,
                'p_value': 0.03,
                'drift_detected': True,
                'severity': 'medium'
            },
        }
    }
    
    # Generate reports
    html_path = generate_html_report(drift_results)
    json_path = generate_json_summary(drift_results)
    
    print(f"\nReports generated:")
    print(f"  HTML: {html_path}")
    print(f"  JSON: {json_path}")
