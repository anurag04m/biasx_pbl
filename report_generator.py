"""
Generate comprehensive bias detection reports
"""

from datetime import datetime
from typing import Dict, Any
import plotly.graph_objects as go

def generate_report(results: Dict, metrics_info: Dict, config: Dict) -> str:
    """
    Generate a comprehensive text report
    
    Args:
        results: Calculation results
        metrics_info: Metric definitions
        config: Configuration details
        
    Returns:
        Formatted text report
    """
    report_lines = []
    
    # Header
    report_lines.append("=" * 80)
    report_lines.append("AIF360 BIAS DETECTION REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Detection Type: {config['detection_type']}")
    report_lines.append("")
    
    # Configuration
    report_lines.append("-" * 80)
    report_lines.append("CONFIGURATION")
    report_lines.append("-" * 80)
    report_lines.append(f"Protected Attribute: {config['protected_attr']}")
    report_lines.append(f"Label Column: {config['label_column']}")
    report_lines.append(f"Privileged Group Value: {config['privileged_value']}")
    report_lines.append(f"Unprivileged Group Value: {config['unprivileged_value']}")
    report_lines.append("")
    
    # Summary
    summary = results['summary']
    report_lines.append("-" * 80)
    report_lines.append("SUMMARY STATISTICS")
    report_lines.append("-" * 80)
    report_lines.append(f"Total Samples: {summary['total_samples']}")
    report_lines.append(f"Privileged Group: {summary['privileged_count']} ({summary['privileged_pct']:.2f}%)")
    report_lines.append(f"Unprivileged Group: {summary['unprivileged_count']} ({summary['unprivileged_pct']:.2f}%)")
    report_lines.append(f"Overall Bias Status: {'BIAS DETECTED' if summary['bias_detected'] else 'FAIR'}")
    report_lines.append("")
    
    # Detailed Metrics
    report_lines.append("-" * 80)
    report_lines.append("DETAILED METRICS ANALYSIS")
    report_lines.append("-" * 80)
    report_lines.append("")
    
    for metric_key, metric_result in results['metrics'].items():
        metric_info = metrics_info[metric_key]
        
        report_lines.append(f"Metric: {metric_info['name']}")
        report_lines.append(f"  Value: {metric_result['value']:.6f}")
        report_lines.append(f"  Status: {'FAIR' if metric_result['is_fair'] else 'BIASED'}")
        report_lines.append(f"  Description: {metric_info['description']}")
        report_lines.append(f"  Formula: {metric_info['formula']}")
        report_lines.append(f"  Ideal Value: {metric_info['ideal_value']}")
        report_lines.append(f"  Interpretation: {metric_result['interpretation']}")
        report_lines.append("")
    
    # Recommendations
    report_lines.append("-" * 80)
    report_lines.append("RECOMMENDATIONS")
    report_lines.append("-" * 80)
    
    biased_metrics = [k for k, v in results['metrics'].items() if not v['is_fair']]
    
    if not biased_metrics:
        report_lines.append("✓ No significant bias detected across selected metrics.")
        report_lines.append("  Continue monitoring fairness metrics regularly.")
    else:
        report_lines.append(f"⚠ {len(biased_metrics)} metric(s) indicate potential bias:")
        for metric_key in biased_metrics:
            report_lines.append(f"  - {metrics_info[metric_key]['name']}")
        
        report_lines.append("")
        report_lines.append("Suggested Actions:")
        report_lines.append("  1. Review data collection process for potential biases")
        report_lines.append("  2. Consider bias mitigation techniques (pre/in/post-processing)")
        report_lines.append("  3. Investigate feature correlations with protected attributes")
        report_lines.append("  4. Consult with domain experts and stakeholders")
        report_lines.append("  5. Consider using AIF360 bias mitigation algorithms")
    
    report_lines.append("")
    report_lines.append("-" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("-" * 80)
    
    return "\n".join(report_lines)

def create_visualizations(results: Dict, metrics_info: Dict) -> Dict[str, go.Figure]:
    """
    Create all visualizations for the report
    
    Args:
        results: Calculation results
        metrics_info: Metric definitions
        
    Returns:
        Dictionary of plotly figures
    """
    figures = {}
    
    # Figure 1: Group Distribution
    figures['group_distribution'] = create_group_distribution(results)
    
    # Figure 2: Metric Values
    figures['metric_values'] = create_metric_values(results, metrics_info)
    
    # Figure 3: Fairness Score
    figures['fairness_score'] = create_fairness_score(results)
    
    return figures

def create_group_distribution(results: Dict) -> go.Figure:
    """Create group distribution chart"""
    summary = results['summary']
    
    fig = go.Figure(data=[
        go.Bar(
            x=['Privileged Group', 'Unprivileged Group'],
            y=[summary['privileged_count'], summary['unprivileged_count']],
            marker_color=['#2ecc71', '#e74c3c'],
            text=[f"{summary['privileged_pct']:.1f}%", f"{summary['unprivileged_pct']:.1f}%"],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Protected Attribute Group Distribution",
        xaxis_title="Group",
        yaxis_title="Sample Count",
        height=400
    )
    
    return fig

def create_metric_values(results: Dict, metrics_info: Dict) -> go.Figure:
    """Create metric values comparison chart"""
    metric_names = []
    metric_values = []
    colors = []
    
    for metric_key, metric_result in results['metrics'].items():
        metric_names.append(metrics_info[metric_key]['name'])
        
        # Use absolute value for visualization clarity
        value = metric_result['value']
        if metric_key not in ['disparate_impact', 'consistency']:
            value = abs(value)
        
        metric_values.append(value)
        colors.append('#2ecc71' if metric_result['is_fair'] else '#e74c3c')
    
    fig = go.Figure(data=[
        go.Bar(
            y=metric_names,
            x=metric_values,
            orientation='h',
            marker_color=colors,
            text=[f"{v:.4f}" for v in metric_values],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Fairness Metrics Comparison",
        xaxis_title="Metric Value",
        yaxis_title="Metric",
        height=max(400, len(metric_names) * 50)
    )
    
    return fig

def create_fairness_score(results: Dict) -> go.Figure:
    """Create overall fairness score pie chart"""
    total_metrics = len(results['metrics'])
    fair_count = sum(1 for m in results['metrics'].values() if m['is_fair'])
    biased_count = total_metrics - fair_count
    
    fig = go.Figure(data=[
        go.Pie(
            labels=['Fair Metrics', 'Biased Metrics'],
            values=[fair_count, biased_count],
            marker=dict(colors=['#2ecc71', '#e74c3c']),
            hole=0.4,
            textinfo='label+value+percent'
        )
    ])
    
    fig.update_layout(
        title=f"Overall Fairness Assessment ({fair_count}/{total_metrics} metrics fair)",
        height=400
    )
    
    return fig
