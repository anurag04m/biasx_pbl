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
    print("[UPLOAD] Received upload request")
    if 'dataset' not in request.files:
        print("[UPLOAD] ERROR: No dataset file in request.files")
        return jsonify({'error': 'No dataset file uploaded (field name must be "dataset")'}), 400
    f = request.files['dataset']
    print(f"[UPLOAD] File received: {f.filename}")
    try:
        df = pd.read_csv(f)
        print(f"[UPLOAD] CSV loaded successfully. Shape: {df.shape}")
    except Exception as e:
        print(f"[UPLOAD] ERROR reading CSV: {str(e)}")
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

    print(f"[UPLOAD] Session created: {session_id}, columns: {columns}")
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
    print("[ANALYZE] Received analyze request")
    payload = request.get_json()
    if payload is None:
        print("[ANALYZE] ERROR: No JSON body")
        return jsonify({'error': 'Expected JSON body'}), 400

    print(f"[ANALYZE] Payload: {payload}")
    sid = payload.get('session_id')
    if not sid or sid not in SESSIONS:
        print(f"[ANALYZE] ERROR: Invalid session_id: {sid}")
        return jsonify({'error': 'Invalid or missing session_id'}), 400

    df = pd.read_csv(io.StringIO(SESSIONS[sid]['df_csv']))
    print(f"[ANALYZE] Loaded dataset from session. Shape: {df.shape}")

    # required config
    required = ['protected_attr', 'label_column', 'privileged_value', 'unprivileged_value', 'selected_metrics']
    for r in required:
        if r not in payload:
            print(f"[ANALYZE] ERROR: Missing parameter: {r}")
            return jsonify({'error': f'Missing required parameter: {r}'}), 400

    protected_attr = payload['protected_attr']
    label_column = payload['label_column']
    privileged_value = payload['privileged_value']
    unprivileged_value = payload['unprivileged_value']
    selected_metrics = payload['selected_metrics']
    detection_type = payload.get('detection_type', 'Dataset Bias Detection')

    print(f"[ANALYZE] Config: protected_attr={protected_attr}, label={label_column}, priv={privileged_value}, unpriv={unprivileged_value}, metrics={selected_metrics}, type={detection_type}")

    # Optional dataset_pred (CSV string) for model bias detection
    dataset_pred = None
    id_column = payload.get('id_column', None)
    pred_label_col = payload.get('pred_label_col', None)
    pred_proba_col = payload.get('pred_proba_col', None)
    proba_threshold = payload.get('proba_threshold', 0.5)

    if 'dataset_pred' in payload and payload['dataset_pred']:
        try:
            dataset_pred = pd.read_csv(io.StringIO(payload['dataset_pred']))
            print(f"[ANALYZE] Loaded dataset_pred. Shape: {dataset_pred.shape}")
            print(f"[ANALYZE] Prediction params: id_col={id_column}, pred_label={pred_label_col}, pred_proba={pred_proba_col}, threshold={proba_threshold}")
        except Exception as e:
            print(f"[ANALYZE] ERROR parsing dataset_pred: {e}")
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
            detection_type=detection_type,
            id_column=id_column,
            pred_label_col=pred_label_col,
            pred_proba_col=pred_proba_col,
            proba_threshold=float(proba_threshold) if proba_threshold else 0.5
        )
        print("[ANALYZE] BiasDetector instantiated successfully")
        results = detector.calculate_metrics(selected_metrics)
        print(f"[ANALYZE] Metrics calculated: {list(results.get('metrics', {}).keys())}")
    except Exception as e:
        import traceback
        print(f"[ANALYZE] ERROR during analysis: {str(e)}")
        print(traceback.format_exc())
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

    print(f"[ANALYZE] Success! Returning results for session {sid}")
    return jsonify({'session_id': sid, 'results': results})

@app.route('/mitigate', methods=['POST'])
def mitigate():
    """Apply dataset mitigation. JSON body must contain session_id and method and optional kwargs.
    Returns mitigated results and a CSV download token.
    """
    print("[MITIGATE] Received mitigation request")
    payload = request.get_json()
    if payload is None:
        print("[MITIGATE] ERROR: No JSON body")
        return jsonify({'error': 'Expected JSON body'}), 400

    print(f"[MITIGATE] Payload: {payload}")
    sid = payload.get('session_id')
    if not sid or sid not in SESSIONS:
        print(f"[MITIGATE] ERROR: Invalid session_id: {sid}")
        return jsonify({'error': 'Invalid or missing session_id'}), 400
    method = payload.get('method')
    if not method:
        print("[MITIGATE] ERROR: Missing method")
        return jsonify({'error': 'Missing mitigation method'}), 400
    kwargs = payload.get('kwargs', {})

    # reload dataset and config
    df = pd.read_csv(io.StringIO(SESSIONS[sid]['df_csv']))
    print(f"[MITIGATE] Loaded dataset. Shape: {df.shape}")
    cfg = SESSIONS[sid].get('config')
    if cfg is None:
        print("[MITIGATE] ERROR: No config found")
        return jsonify({'error': 'No analysis config found for session. Run /analyze first.'}), 400

    print(f"[MITIGATE] Config: {cfg}")
    print(f"[MITIGATE] Method: {method}, kwargs: {kwargs}")

    try:
        detector = BiasDetector(
            dataset=df,
            protected_attr=cfg['protected_attr'],
            label_column=cfg['label_column'],
            privileged_value=cfg['privileged_value'],
            unprivileged_value=cfg['unprivileged_value'],
            detection_type='Dataset Bias Detection'
        )
        print("[MITIGATE] BiasDetector created, applying mitigation...")
        mitigated_df = detector.mitigate_dataset(method, **kwargs)
        print(f"[MITIGATE] Mitigation complete. New shape: {mitigated_df.shape}")
    except Exception as e:
        import traceback
        print(f"[MITIGATE] ERROR during mitigation: {str(e)}")
        print(traceback.format_exc())
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
        print(f"[MITIGATE] Recalculating metrics: {metric_keys}")
        new_results = new_detector.calculate_metrics(metric_keys)
        print(f"[MITIGATE] Recalculation complete")
    except Exception as e:
        import traceback
        print(f"[MITIGATE] ERROR during recalculation: {e}")
        print(traceback.format_exc())
        return jsonify({'error': f'Re-calculation after mitigation failed: {e}'}), 500

    # Update session to use mitigated dataset as the current dataset
    SESSIONS[sid]['df_csv'] = mitigated_df.to_csv(index=False)
    SESSIONS[sid]['last_results'] = new_results

    print(f"[MITIGATE] Success! Session {sid} updated with mitigated dataset")
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
