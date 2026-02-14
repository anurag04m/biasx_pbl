"""
Flask backend for the Bias Detection Tool.
Provides endpoints for dataset upload, analysis, mitigation and download.
"""
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
import pandas as pd
import io
import uuid
import json
from bias_calculator import BiasDetector
from metrics_info import DATASET_METRICS, CLASSIFICATION_METRICS

app = Flask(__name__)
CORS(app)

# Simple in-memory session store. For production replace with persistent store.
SESSIONS = {}

# Utility: limit unique values response length
def _unique_vals(series, max_vals=50):
    vals = pd.Series(series.dropna().unique())
    if len(vals) > max_vals:
        return list(vals[:max_vals].astype(str)) + ["...too many unique values..."]
    return list(vals.astype(str))

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Return available metrics metadata for the frontend to render checkboxes."""
    return jsonify({
        'dataset_metrics': DATASET_METRICS,
        'classification_metrics': CLASSIFICATION_METRICS
    })

@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    """Accepts multipart/form-data with a file field named 'dataset'.
    Returns session_id, columns, sample rows, and unique values for each column.
    """
    if 'dataset' not in request.files:
        return jsonify({'error': 'No dataset file uploaded (field name must be "dataset")'}), 400
    f = request.files['dataset']
    try:
        df = pd.read_csv(f)
    except Exception as e:
        return jsonify({'error': f'Failed to read CSV: {str(e)}'}), 400

    session_id = str(uuid.uuid4())
    # store CSV in-memory
    csv_buf = df.to_csv(index=False)
    SESSIONS[session_id] = {
        'df_csv': csv_buf,
        'last_results': None,
        'config': None
    }

    # prepare columns and unique values (limit size)
    columns = df.columns.tolist()
    sample = df.head(5).to_dict(orient='records')
    uniques = {col: _unique_vals(df[col]) for col in columns}

    return jsonify({
        'session_id': session_id,
        'columns': columns,
        'sample': sample,
        'uniques': uniques
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze the dataset stored in session. Payload JSON should contain:
    - session_id
    - detection_type ('Dataset Bias Detection' or 'Model Bias Detection')
    - protected_attr, label_column, privileged_value, unprivileged_value
    - selected_metrics: list of metric keys
    Optionally include dataset_pred (CSV text) for model detection.
    """
    payload = request.get_json()
    if payload is None:
        return jsonify({'error': 'Expected JSON body'}), 400
    sid = payload.get('session_id')
    if not sid or sid not in SESSIONS:
        return jsonify({'error': 'Invalid or missing session_id'}), 400

    df = pd.read_csv(io.StringIO(SESSIONS[sid]['df_csv']))

    # required config
    required = ['protected_attr', 'label_column', 'privileged_value', 'unprivileged_value', 'selected_metrics']
    for r in required:
        if r not in payload:
            return jsonify({'error': f'Missing required parameter: {r}'}), 400

    protected_attr = payload['protected_attr']
    label_column = payload['label_column']
    privileged_value = payload['privileged_value']
    unprivileged_value = payload['unprivileged_value']
    selected_metrics = payload['selected_metrics']
    detection_type = payload.get('detection_type', 'Dataset Bias Detection')

    # Optional dataset_pred (CSV string) for model bias detection
    dataset_pred = None
    if 'dataset_pred' in payload and payload['dataset_pred']:
        try:
            dataset_pred = pd.read_csv(io.StringIO(payload['dataset_pred']))
        except Exception as e:
            return jsonify({'error': f'Failed to parse dataset_pred CSV: {e}'}), 400

    # Instantiate detector and calculate metrics
    try:
        detector = BiasDetector(
            dataset=df,
            protected_attr=protected_attr,
            label_column=label_column,
            privileged_value=privileged_value,
            unprivileged_value=unprivileged_value,
            dataset_pred=dataset_pred,
            detection_type=detection_type
        )
        results = detector.calculate_metrics(selected_metrics)
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

    # Save config + last_results and last_dataset in session
    SESSIONS[sid]['config'] = {
        'protected_attr': protected_attr,
        'label_column': label_column,
        'privileged_value': privileged_value,
        'unprivileged_value': unprivileged_value,
        'detection_type': detection_type
    }
    SESSIONS[sid]['last_results'] = results

    return jsonify({'session_id': sid, 'results': results})

@app.route('/mitigate', methods=['POST'])
def mitigate():
    """Apply dataset mitigation. JSON body must contain session_id and method and optional kwargs.
    Returns mitigated results and a CSV download token.
    """
    payload = request.get_json()
    if payload is None:
        return jsonify({'error': 'Expected JSON body'}), 400
    sid = payload.get('session_id')
    if not sid or sid not in SESSIONS:
        return jsonify({'error': 'Invalid or missing session_id'}), 400
    method = payload.get('method')
    if not method:
        return jsonify({'error': 'Missing mitigation method'}), 400
    kwargs = payload.get('kwargs', {})

    # reload dataset and config
    df = pd.read_csv(io.StringIO(SESSIONS[sid]['df_csv']))
    cfg = SESSIONS[sid].get('config')
    if cfg is None:
        return jsonify({'error': 'No analysis config found for session. Run /analyze first.'}), 400

    try:
        detector = BiasDetector(
            dataset=df,
            protected_attr=cfg['protected_attr'],
            label_column=cfg['label_column'],
            privileged_value=cfg['privileged_value'],
            unprivileged_value=cfg['unprivileged_value'],
            detection_type='Dataset Bias Detection'
        )
        mitigated_df = detector.mitigate_dataset(method, **kwargs)
    except Exception as e:
        return jsonify({'error': f'Mitigation failed: {str(e)}'}), 500

    # Recalculate metrics on mitigated dataset
    try:
        new_detector = BiasDetector(
            dataset=mitigated_df,
            protected_attr=cfg['protected_attr'],
            label_column=cfg['label_column'],
            privileged_value=cfg['privileged_value'],
            unprivileged_value=cfg['unprivileged_value'],
            detection_type='Dataset Bias Detection'
        )
        metric_keys = list(SESSIONS[sid]['last_results']['metrics'].keys()) if SESSIONS[sid].get('last_results') else list(DATASET_METRICS.keys())
        new_results = new_detector.calculate_metrics(metric_keys)
    except Exception as e:
        return jsonify({'error': f'Re-calculation after mitigation failed: {e}'}), 500

    # Update session to use mitigated dataset as the current dataset
    SESSIONS[sid]['df_csv'] = mitigated_df.to_csv(index=False)
    SESSIONS[sid]['last_results'] = new_results

    # Return new results and provide CSV as a downloadable attachment via /download_dataset
    return jsonify({'session_id': sid, 'new_results': new_results, 'download_endpoint': f'/download_dataset?session_id={sid}&which=mitigated'})

@app.route('/download_dataset', methods=['GET'])
def download_dataset():
    """Return CSV for the current session dataset (which will be the mitigated one if mitigation ran).
    Query params: session_id, which=original|mitigated (original not implemented - returns current)
    """
    sid = request.args.get('session_id')
    if not sid or sid not in SESSIONS:
        return jsonify({'error': 'Invalid or missing session_id'}), 400

    # Return current dataset CSV (the session keeps the most recent CSV)
    csv_text = SESSIONS[sid]['df_csv']
    return Response(
        csv_text,
        mimetype='text/csv',
        headers={"Content-disposition": f"attachment; filename=dataset_{sid}.csv"}
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
