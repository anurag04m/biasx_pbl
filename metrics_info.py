"""
Comprehensive information about AIF360 fairness metrics
"""

DATASET_METRICS = {
    'statistical_parity_difference': {
        'name': 'Statistical Parity Difference',
        'description': 'Measures the difference in selection rates between unprivileged and privileged groups.',
        'formula': 'P(Ŷ=1|D=unprivileged) - P(Ŷ=1|D=privileged)',
        'ideal_value': '0 (no difference)',
        'interpretation': 'A value close to 0 indicates fairness. Negative values indicate the unprivileged group has lower selection rate. Positive values indicate the privileged group has lower selection rate.',
        'threshold': {'min': -0.1, 'max': 0.1},
        'type': 'dataset'
    },
    'disparate_impact': {
        'name': 'Disparate Impact',
        'description': 'Ratio of selection rates between unprivileged and privileged groups. The "80% rule" suggests values should be between 0.8 and 1.25.',
        'formula': 'P(Ŷ=1|D=unprivileged) / P(Ŷ=1|D=privileged)',
        'ideal_value': '1.0 (equal rates)',
        'interpretation': 'Values between 0.8 and 1.25 are generally considered fair. Values < 0.8 indicate discrimination against unprivileged group. Values > 1.25 indicate discrimination against privileged group.',
        'threshold': {'min': 0.8, 'max': 1.25},
        'type': 'dataset'
    },
    'consistency': {
        'name': 'Individual Fairness Consistency',
        'description': 'Measures how similar labels are for k-nearest neighbors. Higher values indicate more consistency and individual fairness.',
        'formula': '1 - (Σ|y_i - ŷ_i|) / n',
        'ideal_value': '1.0 (perfect consistency)',
        'interpretation': 'Values closer to 1.0 indicate better individual fairness. Low values suggest similar individuals are treated differently.',
        'threshold': {'min': 0.8, 'max': 1.0},
        'type': 'dataset'
    },
    'base_rate': {
        'name': 'Base Rate (Positive Rate)',
        'description': 'The proportion of favorable outcomes in the dataset for each group.',
        'formula': 'P(Y=1) for each group',
        'ideal_value': 'Similar across groups',
        'interpretation': 'Large differences in base rates between groups may indicate underlying bias in the data.',
        'threshold': {'min': -0.1, 'max': 0.1},
        'type': 'dataset'
    },
    # Additional dataset-level metrics
    'selection_rate': {
        'name': 'Selection Rate (Statistical Parity)',
        'description': 'Difference in selection rates between unprivileged and privileged groups (same as Statistical Parity Difference).',
        'formula': 'P(Y=1|D=unprivileged) - P(Y=1|D=privileged)',
        'ideal_value': '0 (no difference)',
        'interpretation': 'A value close to 0 indicates similar selection rates across groups. Use this to quickly assess group-level disparities in labels.',
        'threshold': {'min': -0.1, 'max': 0.1},
        'type': 'dataset'
    },
    'positive_rate_ratio': {
        'name': 'Positive Rate Ratio (Disparate Impact Ratio)',
        'description': 'Ratio of selection rates between unprivileged and privileged groups (alternate name for Disparate Impact). Often used with the 80% rule.',
        'formula': 'P(Y=1|D=unprivileged) / P(Y=1|D=privileged)',
        'ideal_value': '1.0 (equal rates)',
        'interpretation': 'Values between 0.8 and 1.25 are commonly considered acceptable. Values outside this range indicate potential disparate impact.',
        'threshold': {'min': 0.8, 'max': 1.25},
        'type': 'dataset'
    }
}

CLASSIFICATION_METRICS = {
    'equal_opportunity_difference': {
        'name': 'Equal Opportunity Difference',
        'description': 'Difference in True Positive Rates (TPR) between groups. Ensures both groups have equal chance of favorable outcomes when they should receive them.',
        'formula': 'TPR_unprivileged - TPR_privileged',
        'ideal_value': '0 (equal TPR)',
        'interpretation': 'A value close to 0 indicates equal opportunity. Negative values mean unprivileged group has lower TPR (missed opportunities). Positive values mean privileged group has lower TPR.',
        'threshold': {'min': -0.1, 'max': 0.1},
        'type': 'classification'
    },
    'average_odds_difference': {
        'name': 'Average Odds Difference',
        'description': 'Average of TPR difference and FPR difference between groups. Comprehensive measure of prediction fairness.',
        'formula': '0.5 * [(TPR_unpriv - TPR_priv) + (FPR_unpriv - FPR_priv)]',
        'ideal_value': '0 (equal odds)',
        'interpretation': 'Value close to 0 indicates fair predictions. Accounts for both true positives and false positives across groups.',
        'threshold': {'min': -0.1, 'max': 0.1},
        'type': 'classification'
    },
    'statistical_parity_difference': {
        'name': 'Statistical Parity Difference (Predictions)',
        'description': 'Difference in predicted positive rates between groups. Similar to dataset metric but for model predictions.',
        'formula': 'P(Ŷ=1|D=unprivileged) - P(Ŷ=1|D=privileged)',
        'ideal_value': '0 (no difference)',
        'interpretation': 'A value close to 0 indicates fair prediction rates. Negative values indicate unprivileged group receives fewer positive predictions.',
        'threshold': {'min': -0.1, 'max': 0.1},
        'type': 'classification'
    },
    'disparate_impact': {
        'name': 'Disparate Impact (Predictions)',
        'description': 'Ratio of predicted positive rates between groups. "80% rule" applies here too.',
        'formula': 'P(Ŷ=1|D=unprivileged) / P(Ŷ=1|D=privileged)',
        'ideal_value': '1.0 (equal rates)',
        'interpretation': 'Values between 0.8 and 1.25 are generally fair. Values outside this range indicate discriminatory predictions.',
        'threshold': {'min': 0.8, 'max': 1.25},
        'type': 'classification'
    },
    'false_positive_rate_difference': {
        'name': 'False Positive Rate Difference',
        'description': 'Difference in FPR between groups. Ensures both groups have equal probability of being incorrectly labeled as positive.',
        'formula': 'FPR_unprivileged - FPR_privileged',
        'ideal_value': '0 (equal FPR)',
        'interpretation': 'Value close to 0 indicates fairness. Positive values mean unprivileged group faces more false accusations.',
        'threshold': {'min': -0.1, 'max': 0.1},
        'type': 'classification'
    },
    'false_negative_rate_difference': {
        'name': 'False Negative Rate Difference',
        'description': 'Difference in FNR between groups. Ensures both groups have equal probability of being incorrectly labeled as negative.',
        'formula': 'FNR_unprivileged - FNR_privileged',
        'ideal_value': '0 (equal FNR)',
        'interpretation': 'Value close to 0 indicates fairness. Positive values mean unprivileged group faces more missed opportunities.',
        'threshold': {'min': -0.1, 'max': 0.1},
        'type': 'classification'
    },
    'false_discovery_rate_difference': {
        'name': 'False Discovery Rate Difference',
        'description': 'Difference in precision between groups. Measures fairness in positive predictive value.',
        'formula': 'FDR_unprivileged - FDR_privileged',
        'ideal_value': '0 (equal precision)',
        'interpretation': 'Value close to 0 indicates similar precision. Non-zero values indicate one group has less reliable positive predictions.',
        'threshold': {'min': -0.1, 'max': 0.1},
        'type': 'classification'
    },
    'false_omission_rate_difference': {
        'name': 'False Omission Rate Difference',
        'description': 'Difference in negative predictive value between groups.',
        'formula': 'FOR_unprivileged - FOR_privileged',
        'ideal_value': '0 (equal NPV)',
        'interpretation': 'Value close to 0 indicates similar negative predictive value. Non-zero values indicate unfairness in negative predictions.',
        'threshold': {'min': -0.1, 'max': 0.1},
        'type': 'classification'
    },
    'error_rate_difference': {
        'name': 'Error Rate Difference',
        'description': 'Difference in overall error rates between groups. Measures accuracy fairness.',
        'formula': 'Error_rate_unprivileged - Error_rate_privileged',
        'ideal_value': '0 (equal accuracy)',
        'interpretation': 'Value close to 0 indicates similar prediction accuracy. Non-zero values mean the model works better for one group.',
        'threshold': {'min': -0.1, 'max': 0.1},
        'type': 'classification'
    },
    'theil_index': {
        'name': 'Theil Index',
        'description': 'Generalized entropy index measuring inequality in predictions. Considers both individual and group fairness.',
        'formula': 'Generalized Entropy with α=1',
        'ideal_value': '0 (perfect equality)',
        'interpretation': 'Lower values indicate more fairness. Higher values indicate greater inequality in benefit allocation.',
        'threshold': {'min': 0, 'max': 0.1},
        'type': 'classification'
    },

    # New metrics added below
    'predictive_parity_difference': {
        'name': 'Predictive Parity Difference',
        'description': 'Difference in Positive Predictive Value (precision) between unprivileged and privileged groups.',
        'formula': 'PPV_unprivileged - PPV_privileged',
        'ideal_value': '0 (equal PPV)',
        'interpretation': 'A value close to 0 indicates predictive parity. Negative values indicate unprivileged group has lower precision.',
        'threshold': {'min': -0.1, 'max': 0.1},
        'type': 'classification'
    },
    'selection_rate_difference': {
        'name': 'Selection Rate Difference (Predictions)',
        'description': 'Difference in predicted positive (selection) rates between groups (prediction-level statistical parity).',
        'formula': 'P(Ŷ=1|D=unprivileged) - P(Ŷ=1|D=privileged)',
        'ideal_value': '0 (no difference)',
        'interpretation': 'A value close to 0 indicates similar selection rates in predictions between groups.',
        'threshold': {'min': -0.1, 'max': 0.1},
        'type': 'classification'
    },
    'balanced_accuracy_difference': {
        'name': 'Balanced Accuracy Difference',
        'description': 'Difference in balanced accuracy (0.5*(TPR+TNR)) between unprivileged and privileged groups.',
        'formula': 'BalancedAcc_unprivileged - BalancedAcc_privileged',
        'ideal_value': '0 (equal balanced accuracy)',
        'interpretation': 'A value close to 0 indicates similar balanced accuracy across groups. Large deviations suggest fairness issues in sensitivity/specificity balance.',
        'threshold': {'min': -0.1, 'max': 0.1},
        'type': 'classification'
    }
}

def get_metric_explanation(metric_key, metric_type='dataset'):
    """Get detailed explanation for a specific metric"""
    if metric_type == 'dataset':
        return DATASET_METRICS.get(metric_key, {})
    else:
        return CLASSIFICATION_METRICS.get(metric_key, {})
