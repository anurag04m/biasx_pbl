import pandas as pd
from bias_calculator import BiasDetector

print('Loading sample_dataset.csv')
df = pd.read_csv('sample_dataset.csv')
print('Shape:', df.shape)
print('Value counts gender:', df['gender'].value_counts().to_dict())

protected='gender'
label='income'
priv=1
unpriv=0

print('Creating BiasDetector...')
det = BiasDetector(dataset=df, protected_attr=protected, label_column=label, privileged_value=priv, unprivileged_value=unpriv)
print('Calculating metrics...')
metrics = det.calculate_metrics(['selection_rate','positive_rate_ratio','base_rate'])
print('Metrics before mitigation:')
print(metrics)

# Try reweighing
print('\nApplying Reweighing...')
try:
    out_rw = det.mitigate_dataset('reweighing')
    print('Reweighing produced instance_weight column? ', 'instance_weight' in out_rw.columns)
    # Recompute metrics using mitigated_df by creating new detector
    det_rw = BiasDetector(dataset=out_rw, protected_attr=protected, label_column=label, privileged_value=priv, unprivileged_value=unpriv)
    metrics_rw = det_rw.calculate_metrics(['selection_rate','positive_rate_ratio','base_rate'])
    print('Metrics after reweighing (dataset metrics):')
    print(metrics_rw)
except Exception as e:
    print('Reweighing failed:', e)

# Try DisparateImpactRemover
print('\nApplying DisparateImpactRemover...')
try:
    out_dir = det.mitigate_dataset('disparate_impact_remover', repair_level=1.0)
    print('DisparateImpactRemover output columns:', out_dir.columns.tolist()[:10])
    print('Value counts gender after DIR:', out_dir['gender'].value_counts().to_dict())
    det_dir = BiasDetector(dataset=out_dir, protected_attr=protected, label_column=label, privileged_value=priv, unprivileged_value=unpriv)
    metrics_dir = det_dir.calculate_metrics(['selection_rate','positive_rate_ratio','base_rate'])
    print('Metrics after DIR:')
    print(metrics_dir)
except Exception as e:
    import traceback
    traceback.print_exc()

# Try Optimized Preprocessing (simple heuristic)
print('\nApplying OptimizedPreprocessing (simple)...')
try:
    out_opt = det.mitigate_dataset('optimized_preprocessing', target_parity=0.0, random_state=42)
    print('Optimized output columns:', out_opt.columns.tolist()[:10])
    print('Value counts gender after Optimized:', out_opt['gender'].value_counts().to_dict())
    det_opt = BiasDetector(dataset=out_opt, protected_attr=protected, label_column=label, privileged_value=priv, unprivileged_value=unpriv)
    metrics_opt = det_opt.calculate_metrics(['selection_rate','positive_rate_ratio','base_rate'])
    print('Metrics after Optimized:')
    print(metrics_opt)
except Exception as e:
    import traceback
    traceback.print_exc()

print('\nDone')
