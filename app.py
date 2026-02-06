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
                # Store detector so we can run mitigation later
                st.session_state.detector = detector
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

            # Safe formatting for display and numeric fallback for charts
            raw_val = metric_result.get('value', None)
            try:
                if isinstance(raw_val, (int, float)) and not (isinstance(raw_val, float) and np.isnan(raw_val)):
                    display_value = f"{raw_val:.4f}"
                    gauge_value = raw_val
                else:
                    display_value = "N/A"
                    gauge_value = 0.0
            except Exception:
                display_value = "N/A"
                gauge_value = 0.0

            with st.expander(f"**{metric_info['name']}** - Value: {display_value}", expanded=True):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(f"**Description:** {metric_info['description']}")
                    st.markdown(f"**Formula:** {metric_info['formula']}")
                    st.markdown(f"**Ideal Value:** {metric_info['ideal_value']}")

                    # Interpretation with color coding
                    if metric_result.get('is_fair'):
                        st.markdown(
                            f'<div class="success-box" style="background-color:#bdecb8; color:#0b3d0b; border-left: 4px solid #28a745; padding:1rem; margin:1rem 0; border-radius:0.5rem;">✅ <strong>Fair:</strong> {metric_result.get("interpretation", "")}</div>',
                            unsafe_allow_html=True)
                    else:
                        st.markdown(
                            f'<div class="danger-box">⚠️ <strong>Biased:</strong> {metric_result.get("interpretation", "")}</div>',
                            unsafe_allow_html=True)

                with col2:
                    # Create gauge chart using the safe numeric value
                    fig = create_gauge_chart(gauge_value, metric_info['name'], metric_key)
                    plot_chart(fig, use_container_width=True)

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

        st.markdown("---")

        # Mitigation Section
        st.header("🛠️ Mitigation Options")

        detector = st.session_state.get('detector')
        if detector is None:
            st.info("Mitigation tools will be available after detection runs successfully.")
        else:
            # Determine which metrics are flagged as biased
            biased = {k: v for k, v in results['metrics'].items() if v.get('is_fair') is False}

            if len(biased) == 0:
                st.success("No significant bias detected across selected metrics — mitigation not required.")
            else:
                st.warning(f"{len(biased)} metric(s) flagged as biased. Suggested mitigations are provided below.")

                # Simple heuristic to recommend mitigation strength
                severity_levels = []
                for k, v in biased.items():
                    val = v.get('value')
                    try:
                        if val is None or (isinstance(val, float) and np.isnan(val)):
                            sev = 0
                        else:
                            # treat disparate impact style ratios separately
                            if k in ('disparate_impact', 'positive_rate_ratio'):
                                dev = abs(float(val) - 1.0)
                                if dev > 0.25:
                                    sev = 3
                                elif dev > 0.1:
                                    sev = 2
                                else:
                                    sev = 1
                            else:
                                # difference metrics: compare to threshold width
                                info = available_metrics.get(k, {})
                                thr = info.get('threshold', {})
                                max_thr = thr.get('max', 0.1)
                                min_thr = thr.get('min', -0.1)
                                # use distance from nearest bound
                                if val < min_thr:
                                    diff = abs(min_thr - float(val))
                                elif val > max_thr:
                                    diff = abs(float(val) - max_thr)
                                else:
                                    diff = 0
                                span = abs(max_thr - min_thr) if (max_thr - min_thr) != 0 else 0.1
                                if diff > 2 * span:
                                    sev = 3
                                elif diff > span:
                                    sev = 2
                                else:
                                    sev = 1
                    except Exception:
                        sev = 1
                    severity_levels.append(sev)

                overall_severity = max(severity_levels) if severity_levels else 0

                # Recommendation logic
                if overall_severity >= 3:
                    recommendation = "High bias detected — recommend trying all mitigation methods (Reweighing, Disparate Impact Remover, Optimized Preprocessing) and evaluating their effects."
                    suggested_methods = ['reweighing', 'disparate_impact_remover', 'optimized_preprocessing']
                elif overall_severity == 2:
                    recommendation = "Moderate bias detected — recommend Disparate Impact Remover and Reweighing as first steps."
                    suggested_methods = ['disparate_impact_remover', 'reweighing']
                else:
                    recommendation = "Minor bias detected — recommend Reweighing (low-cost) first."
                    suggested_methods = ['reweighing']

                st.markdown(f"**Recommendation:** {recommendation}")

                # Present mitigation selection UI
                col_a, col_b = st.columns([2, 1])
                with col_a:
                    method = st.selectbox(
                        "Select Dataset Mitigation Method",
                        options=['reweighing', 'disparate_impact_remover', 'optimized_preprocessing'],
                        format_func=lambda x: x.replace('_', ' ').title(),
                        index=0 if suggested_methods[0] == 'reweighing' else (1 if suggested_methods[0]=='disparate_impact_remover' else 2)
                    )

                    # Additional params for methods
                    repair_level = None
                    if method == 'disparate_impact_remover':
                        repair_level = st.slider("Repair Level", 0.0, 1.0, 1.0, 0.01,
                                                 help="0 = no repair, 1 = full repair")

                    if method == 'optimized_preprocessing':
                        st.info('Optimized Preprocessing can be slow. Only use on small-medium datasets or in experiments.')

                with col_b:
                    st.markdown("**Model Mitigation**")
                    st.markdown("(Placeholder) You can apply post-processing model-level mitigation techniques in your training pipeline.")
                    if st.button("🧩 Apply Model Mitigation (placeholder)"):
                        st.info("Model mitigation UI placeholder created. Backend implementation is not included.")

                # Apply dataset mitigation
                if st.button("▶️ Apply Dataset Mitigation"):
                    try:
                        kwargs = {}
                        if repair_level is not None:
                            kwargs['repair_level'] = repair_level

                        with st.spinner("Applying mitigation — this may take a moment..."):
                            mitigated_df = detector.mitigate_dataset(method, **kwargs)

                        # Save mitigated_df in session for later use
                        st.session_state.mitigated_df = mitigated_df

                        st.success("Mitigation applied successfully. Preview and download below.")

                        with st.expander("📋 Mitigated Dataset Preview", expanded=True):
                            st.dataframe(mitigated_df.head(10))
                            st.write(f"**Shape:** {mitigated_df.shape}")

                        # Offer download
                        csv = mitigated_df.to_csv(index=False)
                        st.download_button(label="⬇️ Download Mitigated Dataset (CSV)", data=csv,
                                           file_name=f"mitigated_dataset_{method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                           mime='text/csv')

                        # Option to re-run metrics on mitigated dataset
                        if st.button("🔁 Re-run Bias Analysis on Mitigated Dataset"):
                            with st.spinner("Re-calculating metrics on mitigated dataset..."):
                                new_detector = BiasDetector(
                                    dataset=mitigated_df,
                                    protected_attr=st.session_state.config['protected_attr'],
                                    label_column=st.session_state.config['label_column'],
                                    privileged_value=st.session_state.config['privileged_value'],
                                    unprivileged_value=st.session_state.config['unprivileged_value'],
                                    detection_type='Dataset Bias Detection'
                                )
                                # Use the same metrics originally selected (or all dataset metrics)
                                metric_keys = list(results['metrics'].keys())
                                new_results = new_detector.calculate_metrics(metric_keys)
                                st.session_state.mitigated_results = new_results

                            st.success("Re-analysis complete. See comparison below.")

                            # Show quick comparison table
                            comp_rows = []
                            for mk in metric_keys:
                                orig = results['metrics'].get(mk, {}).get('value')
                                new = new_results['metrics'].get(mk, {}).get('value')
                                comp_rows.append({'metric': available_metrics[mk]['name'], 'original': orig, 'mitigated': new})

                            comp_df = pd.DataFrame(comp_rows)
                            with st.expander("📊 Mitigation Comparison", expanded=True):
                                st.dataframe(comp_df)

                    except Exception as e:
                        st.error(f"Mitigation failed: {str(e)}")
                        st.exception(e)


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


def plot_chart(fig, use_container_width=True):
    """
    Wrapper around st.plotly_chart:
    - use_container_width=True  -> width='stretch'
    - use_container_width=False -> width='content'
    Falls back to boolean `use_container_width` for older Streamlit versions.
    """
    width = 'stretch' if use_container_width else 'content'
    try:
        st.plotly_chart(fig, width=width)
    except TypeError:
        # Fallback for Streamlit versions that expect use_container_width boolean
        st.plotly_chart(fig, use_container_width=(width == 'stretch'))


def create_metric_comparison_chart(results, available_metrics):
    """Create metric comparison visualization (handles missing values)"""
    metric_names = []
    metric_values = []
    colors = []

    for metric_key, metric_result in results['metrics'].items():
        metric_names.append(available_metrics[metric_key]['name'])
        val = metric_result.get('value', None)
        if isinstance(val, (int, float)) and not (isinstance(val, float) and np.isnan(val)):
            metric_values.append(abs(val))
            colors.append('green' if metric_result.get('is_fair') else 'red')
        else:
            # Show missing values as zero length and gray color
            metric_values.append(0.0)
            colors.append('gray')

    fig = go.Figure(go.Bar(
        x=metric_values,
        y=metric_names,
        orientation='h',
        marker=dict(color=colors),
        text=[(f"{v:.4f}" if v != 0 else "N/A") for v in metric_values],
        textposition='auto'
    ))

    fig.update_layout(
        title="Metric Values Comparison (Absolute)",
        xaxis_title="Absolute Metric Value",
        yaxis_title="Metric",
        height=max(400, len(metric_names) * 50)
    )

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