"""
AIF360 Bias Detection Tool - Streamlit Prototype
Main application file
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO
import sys
from datetime import datetime

# Import custom modules
from metrics_info import DATASET_METRICS, CLASSIFICATION_METRICS, get_metric_explanation
from bias_calculator import BiasDetector
from report_generator import generate_report, create_visualizations

# Page configuration
st.set_page_config(
    page_title="AIF360 Bias Detection Tool",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
    }
    .danger-box {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'dataset_uploaded' not in st.session_state:
        st.session_state.dataset_uploaded = False
    if 'dataset_pred_uploaded' not in st.session_state:
        st.session_state.dataset_pred_uploaded = False
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'report_generated' not in st.session_state:
        st.session_state.report_generated = False


def main():
    initialize_session_state()

    # Header
    st.markdown('<div class="main-header">⚖️ AIF360 Bias Detection Tool</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Detect and analyze fairness in datasets and ML models</div>',
                unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("📊 Configuration")

        # Step 1: Select Detection Type
        st.subheader("1️⃣ Select Detection Type")
        detection_type = st.radio(
            "Choose analysis type:",
            ["Dataset Bias Detection", "Model Bias Detection"],
            help="Dataset: Analyze bias in data. Model: Analyze bias in predictions."
        )

        st.markdown("---")

        # Step 2: Upload Files
        st.subheader("2️⃣ Upload Data")

        uploaded_dataset = st.file_uploader(
            "Upload Dataset (CSV)",
            type=['csv'],
            help="Upload your dataset with features and labels"
        )

        uploaded_pred = None
        if detection_type == "Model Bias Detection":
            uploaded_pred = st.file_uploader(
                "Upload Predictions Dataset (CSV)",
                type=['csv'],
                help="Upload dataset with model predictions"
            )

        st.markdown("---")

        # Step 3: Configure Protected Attributes
        st.subheader("3️⃣ Configure Attributes")

        if uploaded_dataset is not None:
            df = pd.read_csv(uploaded_dataset)
            st.session_state.dataset_uploaded = True

            columns = df.columns.tolist()

            protected_attr = st.selectbox(
                "Protected Attribute",
                columns,
                help="Select the sensitive attribute (e.g., gender, race, age)"
            )

            label_column = st.selectbox(
                "Label Column",
                columns,
                help="Select the target/label column"
            )

            # Define privileged and unprivileged groups
            unique_values = df[protected_attr].unique()

            col1, col2 = st.columns(2)
            with col1:
                privileged_value = st.selectbox(
                    "Privileged Group Value",
                    unique_values,
                    help="Value representing the privileged group"
                )
            with col2:
                unprivileged_value = st.selectbox(
                    "Unprivileged Group Value",
                    [v for v in unique_values if v != privileged_value],
                    help="Value representing the unprivileged group"
                )

            st.session_state.config = {
                'detection_type': detection_type,
                'protected_attr': protected_attr,
                'label_column': label_column,
                'privileged_value': privileged_value,
                'unprivileged_value': unprivileged_value,
                'dataset': df
            }

            if uploaded_pred is not None and detection_type == "Model Bias Detection":
                df_pred = pd.read_csv(uploaded_pred)
                st.session_state.dataset_pred_uploaded = True
                st.session_state.config['dataset_pred'] = df_pred

    # Main content area
    if not st.session_state.dataset_uploaded:
        st.info("👈 Please upload a dataset using the sidebar to get started")

        # Show example format
        with st.expander("📖 Example Dataset Format"):
            st.markdown("""
            Your CSV should have columns like:
            - **Features**: age, education, hours_per_week, etc.
            - **Protected Attribute**: gender, race, etc. (binary: 0/1 or categorical)
            - **Label**: income (0/1 for binary classification)

            Example:
            """)
            example_df = pd.DataFrame({
                'age': [25, 45, 35, 50],
                'education': [12, 16, 14, 18],
                'gender': [0, 1, 0, 1],
                'income': [0, 1, 0, 1]
            })
            st.dataframe(example_df)

        return

    # Show dataset preview
    with st.expander("📋 Dataset Preview", expanded=False):
        st.dataframe(st.session_state.config['dataset'].head(10))
        st.write(f"**Shape:** {st.session_state.config['dataset'].shape}")

    # Metric Selection
    st.header("4️⃣ Select Fairness Metrics")

    if detection_type == "Dataset Bias Detection":
        available_metrics = DATASET_METRICS
        st.info("📊 Select metrics to analyze bias in your dataset")
    else:
        available_metrics = CLASSIFICATION_METRICS
        if not st.session_state.dataset_pred_uploaded:
            st.warning("⚠️ Please upload predictions dataset to proceed with model bias detection")
            return
        st.info("🤖 Select metrics to analyze bias in your model's predictions")

    # Create metric selection interface with explanations
    selected_metrics = []

    cols = st.columns(2)
    for idx, (metric_key, metric_info) in enumerate(available_metrics.items()):
        with cols[idx % 2]:
            with st.container():
                col_check, col_info = st.columns([3, 1])

                with col_check:
                    is_selected = st.checkbox(
                        f"**{metric_info['name']}**",
                        key=f"metric_{metric_key}"
                    )
                    if is_selected:
                        selected_metrics.append(metric_key)

                with col_info:
                    with st.expander("ℹ️"):
                        st.markdown(f"**{metric_info['name']}**")
                        st.markdown(metric_info['description'])
                        st.markdown(f"**Formula:** {metric_info['formula']}")
                        st.markdown(f"**Ideal Value:** {metric_info['ideal_value']}")
                        st.markdown(f"**Interpretation:** {metric_info['interpretation']}")

    st.markdown("---")

    # Generate Report Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        generate_button = st.button(
            "🔍 Detect Bias & Generate Report",
            type="primary",
            use_container_width=True,
            disabled=len(selected_metrics) == 0
        )

    if len(selected_metrics) == 0:
        st.warning("⚠️ Please select at least one metric to generate report")

    # Generate Report
    if generate_button and len(selected_metrics) > 0:
        with st.spinner("🔄 Analyzing bias... This may take a moment..."):
            try:
                # Initialize bias detector
                detector = BiasDetector(
                    dataset=st.session_state.config['dataset'],
                    protected_attr=st.session_state.config['protected_attr'],
                    label_column=st.session_state.config['label_column'],
                    privileged_value=st.session_state.config['privileged_value'],
                    unprivileged_value=st.session_state.config['unprivileged_value'],
                    dataset_pred=st.session_state.config.get('dataset_pred'),
                    detection_type=detection_type
                )

                # Calculate metrics
                results = detector.calculate_metrics(selected_metrics)
                st.session_state.results = results
                st.session_state.report_generated = True

            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.exception(e)
                return

    # Display Results
    if st.session_state.report_generated and st.session_state.results:
        st.success("✅ Bias analysis complete!")

        results = st.session_state.results

        # Summary Section
        st.header("📊 Bias Detection Summary")

        # Key Metrics Display
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("Total Samples", results['summary']['total_samples'])
        with metric_cols[1]:
            st.metric("Privileged Group",
                      f"{results['summary']['privileged_count']} ({results['summary']['privileged_pct']:.1f}%)")
        with metric_cols[2]:
            st.metric("Unprivileged Group",
                      f"{results['summary']['unprivileged_count']} ({results['summary']['unprivileged_pct']:.1f}%)")
        with metric_cols[3]:
            bias_detected = results['summary']['bias_detected']
            st.metric("Bias Status", "⚠️ Detected" if bias_detected else "✅ Fair")

        st.markdown("---")

        # Detailed Metrics
        st.header("📈 Detailed Metrics Analysis")

        for metric_key, metric_result in results['metrics'].items():
            metric_info = available_metrics[metric_key]

            with st.expander(f"**{metric_info['name']}** - Value: {metric_result['value']:.4f}", expanded=True):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(f"**Description:** {metric_info['description']}")
                    st.markdown(f"**Formula:** {metric_info['formula']}")
                    st.markdown(f"**Ideal Value:** {metric_info['ideal_value']}")

                    # Interpretation with color coding
                    if metric_result['is_fair']:
                        st.markdown(
                            f'<div class="success-box" style="background-color:#bdecb8; color:#0b3d0b; border-left: 4px solid #28a745; padding:1rem; margin:1rem 0; border-radius:0.5rem;">✅ <strong>Fair:</strong> {metric_result["interpretation"]}</div>',
                            unsafe_allow_html=True)
                    else:
                        st.markdown(
                            f'<div class="danger-box">⚠️ <strong>Biased:</strong> {metric_result["interpretation"]}</div>',
                            unsafe_allow_html=True)

                with col2:
                    # Create gauge chart
                    fig = create_gauge_chart(
                        metric_result['value'],
                        metric_info['name'],
                        metric_key
                    )
                    st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Visualizations
        st.header("📊 Bias Visualizations")

        viz_tabs = st.tabs(["Group Distribution", "Metric Comparison", "Fairness Dashboard"])

        with viz_tabs[0]:
            fig = create_group_distribution_chart(results)
            st.plotly_chart(fig, use_container_width=True)

        with viz_tabs[1]:
            fig = create_metric_comparison_chart(results, available_metrics)
            st.plotly_chart(fig, use_container_width=True)

        with viz_tabs[2]:
            fig = create_fairness_dashboard(results, available_metrics)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Download Report
        st.header("💾 Export Report")

        col1, col2 = st.columns(2)

        with col1:
            # Generate detailed report
            report_text = generate_report(results, available_metrics, st.session_state.config)
            st.download_button(
                label="📄 Download Text Report",
                data=report_text,
                file_name=f"bias_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

        with col2:
            # Export results as JSON
            import json
            results_json = json.dumps(results, indent=2, default=str)
            st.download_button(
                label="📊 Download JSON Results",
                data=results_json,
                file_name=f"bias_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )


def create_gauge_chart(value, title, metric_key):
    """Create a gauge chart for metric visualization"""
    # Determine range and thresholds based on metric
    if 'disparate_impact' in metric_key:
        # Disparate Impact: ideal around 1.0
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title, 'font': {'size': 12}},
            gauge={
                'axis': {'range': [0, 2]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.8], 'color': "lightcoral"},
                    {'range': [0.8, 1.25], 'color': "lightgreen"},
                    {'range': [1.25, 2], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 1.0
                }
            }
        ))
    else:
        # Difference metrics: ideal around 0
        abs_value = abs(value)
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title, 'font': {'size': 12}},
            delta={'reference': 0},
            gauge={
                'axis': {'range': [-1, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [-1, -0.1], 'color': "lightcoral"},
                    {'range': [-0.1, 0.1], 'color': "lightgreen"},
                    {'range': [0.1, 1], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0
                }
            }
        ))

    fig.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def create_group_distribution_chart(results):
    """Create group distribution visualization"""
    fig = go.Figure()

    # Add bar chart for group sizes
    fig.add_trace(go.Bar(
        name='Group Size',
        x=['Privileged Group', 'Unprivileged Group'],
        y=[results['summary']['privileged_count'], results['summary']['unprivileged_count']],
        marker_color=['#2ecc71', '#e74c3c']
    ))

    fig.update_layout(
        title="Protected Attribute Group Distribution",
        xaxis_title="Group",
        yaxis_title="Count",
        height=400
    )

    return fig


def create_metric_comparison_chart(results, available_metrics):
    """Create metric comparison visualization"""
    metric_names = []
    metric_values = []
    colors = []

    for metric_key, metric_result in results['metrics'].items():
        metric_names.append(available_metrics[metric_key]['name'])
        metric_values.append(abs(metric_result['value']))
        colors.append('green' if metric_result['is_fair'] else 'red')

    fig = go.Figure(go.Bar(
        x=metric_values,
        y=metric_names,
        orientation='h',
        marker=dict(color=colors),
        text=[f"{v:.4f}" for v in metric_values],
        textposition='auto'
    ))

    fig.update_layout(
        title="Metric Values Comparison (Absolute)",
        xaxis_title="Absolute Metric Value",
        yaxis_title="Metric",
        height=max(400, len(metric_names) * 50)
    )

    return fig


def create_fairness_dashboard(results, available_metrics):
    """Create comprehensive fairness dashboard"""
    # Count fair vs biased metrics
    fair_count = sum(1 for m in results['metrics'].values() if m['is_fair'])
    biased_count = len(results['metrics']) - fair_count

    fig = go.Figure(data=[go.Pie(
        labels=['Fair Metrics', 'Biased Metrics'],
        values=[fair_count, biased_count],
        marker=dict(colors=['#2ecc71', '#e74c3c']),
        hole=.3
    )])

    fig.update_layout(
        title="Overall Fairness Assessment",
        height=400,
        annotations=[dict(text=f'{fair_count}/{len(results["metrics"])}', x=0.5, y=0.5, font_size=20, showarrow=False)]
    )

    return fig


if __name__ == "__main__":
    main()