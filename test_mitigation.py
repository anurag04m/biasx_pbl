import pandas as pd
from bias_calculator import BiasDetector

# Small synthetic dataset: 8 samples, protected attribute 'gender' with values 'M' (privileged) and 'F' (unprivileged)
# Label is binary 0/1

data = pd.DataFrame({
    'age': [25, 30, 22, 40, 35, 28, 45, 50],
    'education': [12, 16, 12, 18, 14, 13, 16, 17],
    'gender': ['M','M','F','F','M','F','M','F'],
    'income': [0,1,0,1,1,0,1,0]
})

protected_attr = 'gender'
label_col = 'income'
priv = 'M'
unpriv = 'F'

print('Creating BiasDetector...')
det = BiasDetector(dataset=data, protected_attr=protected_attr, label_column=label_col,
                   privileged_value=priv, unprivileged_value=unpriv,
                   detection_type='Dataset Bias Detection')

print('Calculating metrics...')
res = det.calculate_metrics(['statistical_parity_difference','disparate_impact','base_rate'])
print('Metrics:')
print(res)

# Try reweighing
print('\nTesting Reweighing...')
try:
    out_rw = det.mitigate_dataset('reweighing')
    print('Reweighing output columns:', out_rw.columns.tolist())
    print(out_rw.head())
except Exception as e:
    print('Reweighing failed:', e)

# Try DisparateImpactRemover
print('\nTesting DisparateImpactRemover...')
try:
    out_dir = det.mitigate_dataset('disparate_impact_remover', repair_level=1.0)
    print('DisparateImpactRemover output columns:', out_dir.columns.tolist())
    print(out_dir.head())
except Exception as e:
    print('DisparateImpactRemover failed:', e)

# Try OptimizedPreprocessing
print('\nTesting OptimizedPreprocessing...')
try:
    out_opt = det.mitigate_dataset('optimized_preprocessing')
    print('OptimizedPreprocessing output columns:', out_opt.columns.tolist())
    print(out_opt.head())
except Exception as e:
    print('OptimizedPreprocessing failed:', e)

print('\nTest complete')
