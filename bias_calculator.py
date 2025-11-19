"""
Core bias detection calculation engine
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

class BiasDetector:
    """
    Main class for detecting bias in datasets and model predictions
    """
    
    def __init__(self, dataset: pd.DataFrame, protected_attr: str, label_column: str,
                 privileged_value: Any, unprivileged_value: Any,
                 dataset_pred: Optional[pd.DataFrame] = None,
                 detection_type: str = "Dataset Bias Detection"):
        """
        Initialize BiasDetector
        
        Args:
            dataset: Original dataset
            protected_attr: Name of protected attribute column
            label_column: Name of label column
            privileged_value: Value representing privileged group
            unprivileged_value: Value representing unprivileged group
            dataset_pred: Dataset with predictions (for model bias detection)
            detection_type: Type of detection ("Dataset Bias Detection" or "Model Bias Detection")
        """
        self.dataset = dataset
        self.protected_attr = protected_attr
        self.label_column = label_column
        self.privileged_value = privileged_value
        self.unprivileged_value = unprivileged_value
        self.dataset_pred = dataset_pred
        self.detection_type = detection_type
        
        # Filter dataset for binary groups
        self.privileged_df = dataset[dataset[protected_attr] == privileged_value]
        self.unprivileged_df = dataset[dataset[protected_attr] == unprivileged_value]
        
        if detection_type == "Model Bias Detection" and dataset_pred is not None:
            self.privileged_pred_df = dataset_pred[dataset_pred[protected_attr] == privileged_value]
            self.unprivileged_pred_df = dataset_pred[dataset_pred[protected_attr] == unprivileged_value]
    
    def calculate_metrics(self, selected_metrics: List[str]) -> Dict:
        """
        Calculate selected fairness metrics
        
        Args:
            selected_metrics: List of metric keys to calculate
            
        Returns:
            Dictionary containing results
        """
        results = {
            'summary': self._calculate_summary(),
            'metrics': {}
        }
        
        for metric_key in selected_metrics:
            try:
                if self.detection_type == "Dataset Bias Detection":
                    metric_value = self._calculate_dataset_metric(metric_key)
                else:
                    metric_value = self._calculate_classification_metric(metric_key)
                
                # Determine if metric indicates bias
                is_fair, interpretation = self._interpret_metric(metric_key, metric_value)
                
                results['metrics'][metric_key] = {
                    'value': metric_value,
                    'is_fair': is_fair,
                    'interpretation': interpretation
                }
            except Exception as e:
                results['metrics'][metric_key] = {
                    'value': None,
                    'is_fair': None,
                    'interpretation': f'Error calculating metric: {str(e)}'
                }
        
        # Overall bias detection
        results['summary']['bias_detected'] = any(
            not m['is_fair'] for m in results['metrics'].values() if m['is_fair'] is not None
        )
        
        return results
    
    def _calculate_summary(self) -> Dict:
        """Calculate summary statistics"""
        total = len(self.dataset)
        priv_count = len(self.privileged_df)
        unpriv_count = len(self.unprivileged_df)
        
        return {
            'total_samples': total,
            'privileged_count': priv_count,
            'unprivileged_count': unpriv_count,
            'privileged_pct': (priv_count / total) * 100,
            'unprivileged_pct': (unpriv_count / total) * 100,
            'bias_detected': False  # Will be updated after metric calculation
        }
    
    def _calculate_dataset_metric(self, metric_key: str) -> float:
        """Calculate dataset-level fairness metrics"""
        
        if metric_key == 'statistical_parity_difference':
            return self._statistical_parity_difference(use_predictions=False)
        
        elif metric_key == 'disparate_impact':
            return self._disparate_impact(use_predictions=False)
        
        elif metric_key == 'consistency':
            return self._consistency()
        
        elif metric_key == 'base_rate':
            return self._base_rate_difference()
        
        else:
            raise ValueError(f"Unknown dataset metric: {metric_key}")
    
    def _calculate_classification_metric(self, metric_key: str) -> float:
        """Calculate classification-level fairness metrics"""
        
        if self.dataset_pred is None:
            raise ValueError("Predictions dataset required for classification metrics")
        
        if metric_key == 'equal_opportunity_difference':
            return self._equal_opportunity_difference()
        
        elif metric_key == 'average_odds_difference':
            return self._average_odds_difference()
        
        elif metric_key == 'statistical_parity_difference':
            return self._statistical_parity_difference(use_predictions=True)
        
        elif metric_key == 'disparate_impact':
            return self._disparate_impact(use_predictions=True)
        
        elif metric_key == 'false_positive_rate_difference':
            return self._false_positive_rate_difference()
        
        elif metric_key == 'false_negative_rate_difference':
            return self._false_negative_rate_difference()
        
        elif metric_key == 'false_discovery_rate_difference':
            return self._false_discovery_rate_difference()
        
        elif metric_key == 'false_omission_rate_difference':
            return self._false_omission_rate_difference()
        
        elif metric_key == 'error_rate_difference':
            return self._error_rate_difference()
        
        elif metric_key == 'theil_index':
            return self._theil_index()
        
        else:
            raise ValueError(f"Unknown classification metric: {metric_key}")
    
    # ========== Dataset Metrics ==========
    
    def _statistical_parity_difference(self, use_predictions: bool = False) -> float:
        """Calculate Statistical Parity Difference"""
        if use_predictions:
            priv_positive_rate = (self.privileged_pred_df[self.label_column] == 1).mean()
            unpriv_positive_rate = (self.unprivileged_pred_df[self.label_column] == 1).mean()
        else:
            priv_positive_rate = (self.privileged_df[self.label_column] == 1).mean()
            unpriv_positive_rate = (self.unprivileged_df[self.label_column] == 1).mean()
        
        return unpriv_positive_rate - priv_positive_rate
    
    def _disparate_impact(self, use_predictions: bool = False) -> float:
        """Calculate Disparate Impact"""
        if use_predictions:
            priv_positive_rate = (self.privileged_pred_df[self.label_column] == 1).mean()
            unpriv_positive_rate = (self.unprivileged_pred_df[self.label_column] == 1).mean()
        else:
            priv_positive_rate = (self.privileged_df[self.label_column] == 1).mean()
            unpriv_positive_rate = (self.unprivileged_df[self.label_column] == 1).mean()
        
        if priv_positive_rate == 0:
            return float('inf') if unpriv_positive_rate > 0 else 1.0
        
        return unpriv_positive_rate / priv_positive_rate
    
    def _consistency(self, n_neighbors: int = 5) -> float:
        """Calculate Individual Fairness Consistency"""
        from sklearn.neighbors import NearestNeighbors
        
        # Get features (exclude protected attr and label)
        feature_cols = [col for col in self.dataset.columns 
                       if col not in [self.protected_attr, self.label_column]]
        
        X = self.dataset[feature_cols].values
        y = self.dataset[self.label_column].values
        
        # Fit KNN
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)
        _, indices = nbrs.kneighbors(X)
        
        # Calculate consistency
        consistency_sum = 0
        for i, neighbors in enumerate(indices):
            neighbors = neighbors[1:]  # Exclude self
            consistency_sum += np.sum(y[neighbors] == y[i])
        
        return consistency_sum / (len(y) * n_neighbors)
    
    def _base_rate_difference(self) -> float:
        """Calculate difference in base rates (positive outcome rates)"""
        priv_base_rate = (self.privileged_df[self.label_column] == 1).mean()
        unpriv_base_rate = (self.unprivileged_df[self.label_column] == 1).mean()
        
        return abs(unpriv_base_rate - priv_base_rate)
    
    # ========== Classification Metrics ==========
    
    def _confusion_matrix_rates(self, group_df_original, group_df_pred):
        """Calculate TPR, FPR, TNR, FNR for a group"""
        y_true = group_df_original[self.label_column].values
        y_pred = group_df_pred[self.label_column].values
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate (Recall)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
        
        # Additional rates
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision (Positive Predictive Value)
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        fdr = fp / (fp + tp) if (fp + tp) > 0 else 0  # False Discovery Rate
        for_rate = fn / (fn + tn) if (fn + tn) > 0 else 0  # False Omission Rate
        
        error_rate = (fp + fn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        return {
            'tpr': tpr, 'fpr': fpr, 'tnr': tnr, 'fnr': fnr,
            'ppv': ppv, 'npv': npv, 'fdr': fdr, 'for': for_rate,
            'error_rate': error_rate
        }
    
    def _equal_opportunity_difference(self) -> float:
        """Calculate Equal Opportunity Difference (TPR difference)"""
        priv_rates = self._confusion_matrix_rates(self.privileged_df, self.privileged_pred_df)
        unpriv_rates = self._confusion_matrix_rates(self.unprivileged_df, self.unprivileged_pred_df)
        
        return unpriv_rates['tpr'] - priv_rates['tpr']
    
    def _average_odds_difference(self) -> float:
        """Calculate Average Odds Difference"""
        priv_rates = self._confusion_matrix_rates(self.privileged_df, self.privileged_pred_df)
        unpriv_rates = self._confusion_matrix_rates(self.unprivileged_df, self.unprivileged_pred_df)
        
        tpr_diff = unpriv_rates['tpr'] - priv_rates['tpr']
        fpr_diff = unpriv_rates['fpr'] - priv_rates['fpr']
        
        return 0.5 * (tpr_diff + fpr_diff)
    
    def _false_positive_rate_difference(self) -> float:
        """Calculate FPR Difference"""
        priv_rates = self._confusion_matrix_rates(self.privileged_df, self.privileged_pred_df)
        unpriv_rates = self._confusion_matrix_rates(self.unprivileged_df, self.unprivileged_pred_df)
        
        return unpriv_rates['fpr'] - priv_rates['fpr']
    
    def _false_negative_rate_difference(self) -> float:
        """Calculate FNR Difference"""
        priv_rates = self._confusion_matrix_rates(self.privileged_df, self.privileged_pred_df)
        unpriv_rates = self._confusion_matrix_rates(self.unprivileged_df, self.unprivileged_pred_df)
        
        return unpriv_rates['fnr'] - priv_rates['fnr']
    
    def _false_discovery_rate_difference(self) -> float:
        """Calculate FDR Difference"""
        priv_rates = self._confusion_matrix_rates(self.privileged_df, self.privileged_pred_df)
        unpriv_rates = self._confusion_matrix_rates(self.unprivileged_df, self.unprivileged_pred_df)
        
        return unpriv_rates['fdr'] - priv_rates['fdr']
    
    def _false_omission_rate_difference(self) -> float:
        """Calculate FOR Difference"""
        priv_rates = self._confusion_matrix_rates(self.privileged_df, self.privileged_pred_df)
        unpriv_rates = self._confusion_matrix_rates(self.unprivileged_df, self.unprivileged_pred_df)
        
        return unpriv_rates['for'] - priv_rates['for']
    
    def _error_rate_difference(self) -> float:
        """Calculate Error Rate Difference"""
        priv_rates = self._confusion_matrix_rates(self.privileged_df, self.privileged_pred_df)
        unpriv_rates = self._confusion_matrix_rates(self.unprivileged_df, self.unprivileged_pred_df)
        
        return unpriv_rates['error_rate'] - priv_rates['error_rate']
    
    def _theil_index(self) -> float:
        """Calculate Theil Index (Generalized Entropy Index with α=1)"""
        # Get predictions
        y_pred = self.dataset_pred[self.label_column].values
        
        # Calculate benefit (1 for correct prediction, 0 for incorrect)
        y_true = self.dataset[self.label_column].values
        benefits = (y_pred == y_true).astype(float)
        
        # Calculate mean benefit
        mu = benefits.mean()
        
        if mu == 0:
            return 0.0
        
        # Calculate Theil index
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-10
        benefits_safe = np.where(benefits == 0, epsilon, benefits)
        
        theil = np.mean((benefits_safe / mu) * np.log(benefits_safe / mu))
        
        return theil
    
    def _interpret_metric(self, metric_key: str, value: float) -> tuple:
        """
        Interpret metric value to determine if it indicates bias
        
        Returns:
            (is_fair: bool, interpretation: str)
        """
        from metrics_info import DATASET_METRICS, CLASSIFICATION_METRICS
        
        # Get metric info
        if self.detection_type == "Dataset Bias Detection":
            metric_info = DATASET_METRICS.get(metric_key, {})
        else:
            metric_info = CLASSIFICATION_METRICS.get(metric_key, {})
        
        threshold = metric_info.get('threshold', {})
        
        if metric_key in ['disparate_impact']:
            # Special case: disparate impact ratio
            is_fair = threshold['min'] <= value <= threshold['max']
            if is_fair:
                interpretation = f"The ratio ({value:.4f}) falls within the acceptable range of {threshold['min']}-{threshold['max']}."
            elif value < threshold['min']:
                interpretation = f"The ratio ({value:.4f}) is below {threshold['min']}, indicating the unprivileged group is disadvantaged."
            else:
                interpretation = f"The ratio ({value:.4f}) is above {threshold['max']}, indicating the privileged group is disadvantaged."
        
        elif metric_key in ['consistency']:
            # Higher is better
            is_fair = value >= threshold['min']
            if is_fair:
                interpretation = f"Consistency score ({value:.4f}) is good, indicating similar individuals are treated similarly."
            else:
                interpretation = f"Consistency score ({value:.4f}) is low, suggesting individual fairness concerns."
        
        elif metric_key in ['theil_index']:
            # Lower is better (closer to 0)
            is_fair = value <= threshold['max']
            if is_fair:
                interpretation = f"Theil index ({value:.4f}) is low, indicating good fairness in benefit distribution."
            else:
                interpretation = f"Theil index ({value:.4f}) is high, indicating inequality in outcomes."
        
        else:
            # Difference metrics: should be close to 0
            is_fair = threshold['min'] <= value <= threshold['max']
            if is_fair:
                interpretation = f"The difference ({value:.4f}) is within acceptable bounds ({threshold['min']} to {threshold['max']})."
            elif value < threshold['min']:
                interpretation = f"The difference ({value:.4f}) is significantly negative, indicating the unprivileged group is disadvantaged."
            else:
                interpretation = f"The difference ({value:.4f}) is significantly positive, indicating the privileged group is disadvantaged."
        
        return is_fair, interpretation
