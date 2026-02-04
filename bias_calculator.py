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
                  detection_type: str = "Dataset Bias Detection",
                  id_column: Optional[str] = None,
                  pred_label_col: Optional[str] = None,
                  pred_proba_col: Optional[str] = None,
                  proba_threshold: float = 0.5,
                  min_group_size: int = 30,
                  label_mapping: Optional[Dict[Any, int]] = None):
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
            id_column: Optional column name to join `dataset` and `dataset_pred` on
            pred_label_col: Column name in `dataset_pred` containing predicted class labels (0/1)
            pred_proba_col: Column name in `dataset_pred` containing predicted probabilities for positive class
            proba_threshold: Threshold to convert probabilities to binary predictions (if used)
            min_group_size: Minimum samples per group to consider metric reliable
        """
        self.dataset = dataset.copy()
        self.protected_attr = protected_attr
        self.label_column = label_column
        self.privileged_value = privileged_value
        self.unprivileged_value = unprivileged_value
        self.dataset_pred = dataset_pred.copy() if dataset_pred is not None else None
        self.detection_type = detection_type
        self.id_column = id_column
        self.pred_label_col = pred_label_col
        self.pred_proba_col = pred_proba_col
        self.proba_threshold = proba_threshold
        self.min_group_size = min_group_size
        # Optional mapping for converting arbitrary label values to binary 0/1
        self.label_mapping = label_mapping

        # Basic validation and preparation
        self._validate_inputs()

        # Filter dataset for binary groups
        self.privileged_df = self.dataset[self.dataset[protected_attr] == privileged_value]
        self.unprivileged_df = self.dataset[self.dataset[protected_attr] == unprivileged_value]

        # Prepare prediction DataFrames (aligned to original dataset if provided)
        self.privileged_pred_df = None
        self.unprivileged_pred_df = None
        if detection_type == "Model Bias Detection" and self.dataset_pred is not None:
            # If id_column is provided, merge to align with original dataset rows
            if self.id_column is not None and self.id_column in self.dataset.columns and self.id_column in self.dataset_pred.columns:
                merged = pd.merge(
                    self.dataset[[self.id_column, self.protected_attr, self.label_column]],
                    self.dataset_pred, on=self.id_column, how='inner', suffixes=('_orig', '_pred')
                )
                # If predictions are probabilities, create label column
                if self.pred_label_col is None and self.pred_proba_col is not None and self.pred_proba_col in merged.columns:
                    merged[self.label_column + '_pred'] = (merged[self.pred_proba_col] >= self.proba_threshold).astype(int)
                    self.pred_label_col = self.label_column + '_pred'
                # If pred_label_col specified and exists, ensure it's available under the expected name
                elif self.pred_label_col is not None and self.pred_label_col in merged.columns:
                    # ok
                    pass
                else:
                    # Try to infer predicted label column in dataset_pred
                    possible = [c for c in merged.columns if c.lower() in (self.label_column.lower(), 'y_pred', 'pred', 'prediction')]
                    if possible:
                        self.pred_label_col = possible[0]
                # Set dataset_pred to merged
                self.dataset_pred = merged
            else:
                # No id_column merge; assume dataset_pred has same indexing and required columns
                # If pred_proba_col provided but pred_label_col not, derive labels
                if self.pred_label_col is None and self.pred_proba_col is not None and self.pred_proba_col in self.dataset_pred.columns:
                    self.dataset_pred[self.label_column + '_pred'] = (self.dataset_pred[self.pred_proba_col] >= self.proba_threshold).astype(int)
                    self.pred_label_col = self.label_column + '_pred'
                # Try to infer predicted label column
                if self.pred_label_col is None:
                    possible = [c for c in self.dataset_pred.columns if c.lower() in (self.label_column.lower(), 'y_pred', 'pred', 'prediction')]
                    if possible:
                        self.pred_label_col = possible[0]

            # After ensuring pred_label_col exists, apply optional mapping to predicted labels
            if self.pred_label_col is None or self.pred_label_col not in self.dataset_pred.columns:
                raise ValueError('Predicted labels not found or could not be inferred. Provide `pred_label_col` or `pred_proba_col`.')

            # If user provided a label_mapping, map predicted labels on dataset_pred
            if self.label_mapping is not None:
                # map and ensure no unmapped values
                self.dataset_pred[self.pred_label_col] = self.dataset_pred[self.pred_label_col].map(self.label_mapping)
                if self.dataset_pred[self.pred_label_col].isnull().any():
                    raise ValueError('Some predicted label values were not found in provided label_mapping')
                self.dataset_pred[self.pred_label_col] = self.dataset_pred[self.pred_label_col].astype(int)

            # Now create group-specific prediction DataFrames (aligned to protected attribute)
            if self.protected_attr in self.dataset_pred.columns:
                # predictions already include protected attribute
                self.privileged_pred_df = self.dataset_pred[self.dataset_pred[self.protected_attr] == privileged_value]
                self.unprivileged_pred_df = self.dataset_pred[self.dataset_pred[self.protected_attr] == unprivileged_value]
            else:
                # Try to align via id_column if available
                if self.id_column is not None and self.id_column in self.dataset_pred.columns and self.id_column in self.dataset.columns:
                    merged = pd.merge(
                        self.dataset[[self.id_column, self.protected_attr]],
                        self.dataset_pred[[self.id_column, self.pred_label_col]],
                        on=self.id_column, how='inner'
                    )
                    self.privileged_pred_df = merged[merged[self.protected_attr] == privileged_value]
                    self.unprivileged_pred_df = merged[merged[self.protected_attr] == unprivileged_value]
                else:
                    # As a last resort, assume same ordering and align by index if lengths match
                    if len(self.dataset_pred) == len(self.dataset):
                        temp = self.dataset_pred.copy()
                        temp[self.protected_attr] = self.dataset[self.protected_attr].values
                        self.privileged_pred_df = temp[temp[self.protected_attr] == privileged_value]
                        self.unprivileged_pred_df = temp[temp[self.protected_attr] == unprivileged_value]
                    else:
                        raise ValueError('Unable to align predictions with original dataset; provide `id_column` or include protected attribute in predictions.')

    def _validate_inputs(self) -> None:
        """Validate presence of required columns and basic sanity checks."""
        if self.protected_attr not in self.dataset.columns:
            raise ValueError(f"Protected attribute '{self.protected_attr}' not found in dataset")
        if self.label_column not in self.dataset.columns:
            raise ValueError(f"Label column '{self.label_column}' not found in dataset")
        # Check that privileged and unprivileged groups exist
        if self.privileged_value not in self.dataset[self.protected_attr].unique():
            raise ValueError("Privileged value not found in protected attribute column")
        if self.unprivileged_value not in self.dataset[self.protected_attr].unique():
            raise ValueError("Unprivileged value not found in protected attribute column")
        # Basic label sanity: ensure binary-like labels (0/1 or boolean). Apply label_mapping if provided.
        if self.label_mapping is not None:
            # Map values using provided mapping. Values not in mapping will become NaN.
            self.dataset[self.label_column] = self.dataset[self.label_column].map(self.label_mapping)
            if self.dataset[self.label_column].isnull().any():
                raise ValueError("Some label values in dataset were not found in provided label_mapping")
            # ensure integer type
            self.dataset[self.label_column] = self.dataset[self.label_column].astype(int)
        else:
            unique_labels = pd.Series(self.dataset[self.label_column].unique())
            if not set(unique_labels).issubset({0,1,True,False}):
                # Attempt to coerce common positive labels
                mapping = {}
                if set(unique_labels.astype(str)).issuperset({'yes','no'}):
                    mapping = {'yes':1,'no':0}
                elif set(unique_labels.astype(str)).issuperset({'positive','negative'}):
                    mapping = {'positive':1,'negative':0}
                if mapping:
                    self.dataset[self.label_column] = self.dataset[self.label_column].astype(str).map(mapping).astype(int)
                else:
                    # Leave as-is; downstream code will error if unsupported
                    pass

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
            notes = []
            reliable = True
            metric_value = None
            try:
                # Check group sizes for reliability
                priv_count = len(self.privileged_df)
                unpriv_count = len(self.unprivileged_df)
                if priv_count < self.min_group_size or unpriv_count < self.min_group_size:
                    reliable = False
                    notes.append(f"One or both groups have < {self.min_group_size} samples; metric may be unreliable.")

                if self.detection_type == "Dataset Bias Detection":
                    metric_value = self._calculate_dataset_metric(metric_key)
                else:
                    metric_value = self._calculate_classification_metric(metric_key)

                # Determine if metric indicates bias
                is_fair, interpretation = self._interpret_metric(metric_key, metric_value)

                results['metrics'][metric_key] = {
                    'value': metric_value,
                    'is_fair': is_fair,
                    'interpretation': interpretation,
                    'reliable': reliable,
                    'notes': notes
                }
            except Exception as e:
                results['metrics'][metric_key] = {
                    'value': None,
                    'is_fair': None,
                    'interpretation': f'Error calculating metric: {str(e)}',
                    'reliable': False,
                    'notes': [str(e)]
                }
        
        # Overall bias detection: only consider metrics that returned a boolean is_fair
        results['summary']['bias_detected'] = any(
            (not m['is_fair']) for m in results['metrics'].values() if isinstance(m.get('is_fair'), bool)
        )
        
        return results
    
    def _calculate_summary(self) -> Dict:
        """Calculate summary statistics"""
        total = len(self.dataset)
        priv_count = len(self.dataset[self.protected_attr] == self.privileged_value) if self.protected_attr in self.dataset else 0
        # Note: fix calculation to count occurrences
        priv_count = int((self.dataset[self.protected_attr] == self.privileged_value).sum())
        unpriv_count = int((self.dataset[self.protected_attr] == self.unprivileged_value).sum())

        privileged_pct = (priv_count / total) * 100 if total > 0 else 0
        unprivileged_pct = (unpriv_count / total) * 100 if total > 0 else 0

        return {
            'total_samples': total,
            'privileged_count': priv_count,
            'unprivileged_count': unpriv_count,
            'privileged_pct': privileged_pct,
            'unprivileged_pct': unprivileged_pct,
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

        # New classification metrics
        elif metric_key == 'predictive_parity_difference':
            return self._predictive_parity_difference()

        elif metric_key == 'selection_rate_difference':
            # selection rate difference == statistical parity difference on predictions
            return self._statistical_parity_difference(use_predictions=True)

        elif metric_key == 'balanced_accuracy_difference':
            return self._balanced_accuracy_difference()

        else:
            raise ValueError(f"Unknown classification metric: {metric_key}")
    
    # ========== Dataset Metrics ==========
    
    def _statistical_parity_difference(self, use_predictions: bool = False) -> float:
        """Calculate Statistical Parity Difference"""
        if use_predictions:
            priv_positive_rate = (self.privileged_pred_df[self.pred_label_col] == 1).mean() if (self.privileged_pred_df is not None and self.pred_label_col in self.privileged_pred_df.columns) else np.nan
            unpriv_positive_rate = (self.unprivileged_pred_df[self.pred_label_col] == 1).mean() if (self.unprivileged_pred_df is not None and self.pred_label_col in self.unprivileged_pred_df.columns) else np.nan
        else:
            priv_positive_rate = (self.privileged_df[self.label_column] == 1).mean() if len(self.privileged_df) > 0 else np.nan
            unpriv_positive_rate = (self.unprivileged_df[self.label_column] == 1).mean() if len(self.unprivileged_df) > 0 else np.nan

        if np.isnan(priv_positive_rate) or np.isnan(unpriv_positive_rate):
            return float('nan')

        return unpriv_positive_rate - priv_positive_rate
    
    def _disparate_impact(self, use_predictions: bool = False) -> float:
        """Calculate Disparate Impact"""
        if use_predictions:
            priv_positive_rate = (self.privileged_pred_df[self.pred_label_col] == 1).mean() if (self.privileged_pred_df is not None and self.pred_label_col in self.privileged_pred_df.columns) else np.nan
            unpriv_positive_rate = (self.unprivileged_pred_df[self.pred_label_col] == 1).mean() if (self.unprivileged_pred_df is not None and self.pred_label_col in self.unprivileged_pred_df.columns) else np.nan
        else:
            priv_positive_rate = (self.privileged_df[self.label_column] == 1).mean() if len(self.privileged_df) > 0 else np.nan
            unpriv_positive_rate = (self.unprivileged_df[self.label_column] == 1).mean() if len(self.unprivileged_df) > 0 else np.nan

        if np.isnan(priv_positive_rate) or np.isnan(unpriv_positive_rate):
            return float('nan')

        # Avoid infinite ratios; return nan when denominator is zero and numerator > 0
        if priv_positive_rate == 0:
            return float('nan')

        return unpriv_positive_rate / priv_positive_rate
    
    def _consistency(self, n_neighbors: int = 5) -> float:
        """Calculate Individual Fairness Consistency"""
        from sklearn.neighbors import NearestNeighbors
        
        # Get features (exclude protected attr and label)
        feature_cols = [col for col in self.dataset.columns 
                       if col not in [self.protected_attr, self.label_column]]
        
        X = self.dataset[feature_cols].values
        y = self.dataset[self.label_column].values
        
        if len(y) == 0:
            return float('nan')

        # Fit KNN
        nbrs = NearestNeighbors(n_neighbors=min(n_neighbors + 1, len(y))).fit(X)
        _, indices = nbrs.kneighbors(X)
        
        # Calculate consistency
        consistency_sum = 0
        for i, neighbors in enumerate(indices):
            neighbors = neighbors[1:]
            consistency_sum += np.sum(y[neighbors] == y[i])
        
        return consistency_sum / (len(y) * min(n_neighbors, len(y)-1))

    def _base_rate_difference(self) -> float:
        """Calculate difference in base rates (positive outcome rates)"""
        priv_base_rate = (self.privileged_df[self.label_column] == 1).mean() if len(self.privileged_df) > 0 else np.nan
        unpriv_base_rate = (self.unprivileged_df[self.label_column] == 1).mean() if len(self.unprivileged_df) > 0 else np.nan

        if np.isnan(priv_base_rate) or np.isnan(unpriv_base_rate):
            return float('nan')

        return abs(unpriv_base_rate - priv_base_rate)
    
    # ========== Classification Metrics ==========
    
    def _confusion_matrix_rates(self, group_df_original, group_df_pred):
        """Calculate TPR, FPR, TNR, FNR for a group"""
        # Accept either merged DataFrame with both true and pred columns, or two separate frames
        if group_df_pred is None:
            return None

        # Determine columns for true and predicted labels
        if self.label_column in group_df_original.columns:
            y_true = group_df_original[self.label_column].values
        else:
            raise ValueError(f"True label column '{self.label_column}' not found in original group dataframe")

        # group_df_pred might be a merged frame containing pred column or just pred col
        if self.pred_label_col in group_df_pred.columns:
            y_pred = group_df_pred[self.pred_label_col].values
        elif self.label_column in group_df_pred.columns:
            y_pred = group_df_pred[self.label_column].values
        else:
            # If group_df_pred has single column of predictions, try to use it
            if group_df_pred.shape[1] == 1:
                y_pred = group_df_pred.iloc[:,0].values
            else:
                raise ValueError('Predicted label column not found in predictions dataframe')

        # Ensure matching lengths: if lengths differ, try to align by index
        if len(y_true) != len(y_pred):
            # If group_df_pred contains index or id aligned, attempt alignment
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]

        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        # Use safe divisions; return np.nan for undefined rates
        tpr = tp / (tp + fn) if (tp + fn) > 0 else float('nan')  # True Positive Rate (Recall)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else float('nan')  # False Positive Rate
        tnr = tn / (tn + fp) if (tn + fp) > 0 else float('nan')  # True Negative Rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else float('nan')  # False Negative Rate

        # Additional rates
        ppv = tp / (tp + fp) if (tp + fp) > 0 else float('nan')  # Precision (Positive Predictive Value)
        npv = tn / (tn + fn) if (tn + fn) > 0 else float('nan')  # Negative Predictive Value
        fdr = fp / (fp + tp) if (fp + tp) > 0 else float('nan')  # False Discovery Rate
        for_rate = fn / (fn + tn) if (fn + tn) > 0 else float('nan')  # False Omission Rate

        error_rate = (fp + fn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else float('nan')

        return {
            'tpr': tpr, 'fpr': fpr, 'tnr': tnr, 'fnr': fnr,
            'ppv': ppv, 'npv': npv, 'fdr': fdr, 'for': for_rate,
            'error_rate': error_rate
        }
    
    def _equal_opportunity_difference(self) -> float:
        """Calculate Equal Opportunity Difference (TPR difference)"""
        priv_rates = self._confusion_matrix_rates(self.privileged_df, self.privileged_pred_df)
        unpriv_rates = self._confusion_matrix_rates(self.unprivileged_df, self.unprivileged_pred_df)
        
        if priv_rates is None or unpriv_rates is None:
            return float('nan')

        if np.isnan(priv_rates['tpr']) or np.isnan(unpriv_rates['tpr']):
            return float('nan')

        return unpriv_rates['tpr'] - priv_rates['tpr']
    
    def _average_odds_difference(self) -> float:
        """Calculate Average Odds Difference"""
        priv_rates = self._confusion_matrix_rates(self.privileged_df, self.privileged_pred_df)
        unpriv_rates = self._confusion_matrix_rates(self.unprivileged_df, self.unprivileged_pred_df)
        
        if priv_rates is None or unpriv_rates is None:
            return float('nan')

        tpr_priv = priv_rates.get('tpr', np.nan)
        tpr_unpriv = unpriv_rates.get('tpr', np.nan)
        fpr_priv = priv_rates.get('fpr', np.nan)
        fpr_unpriv = unpriv_rates.get('fpr', np.nan)

        if np.isnan(tpr_priv) or np.isnan(tpr_unpriv) or np.isnan(fpr_priv) or np.isnan(fpr_unpriv):
            return float('nan')

        tpr_diff = tpr_unpriv - tpr_priv
        fpr_diff = fpr_unpriv - fpr_priv

        return 0.5 * (tpr_diff + fpr_diff)
    
    def _false_positive_rate_difference(self) -> float:
        """Calculate FPR Difference"""
        priv_rates = self._confusion_matrix_rates(self.privileged_df, self.privileged_pred_df)
        unpriv_rates = self._confusion_matrix_rates(self.unprivileged_df, self.unprivileged_pred_df)
        
        if priv_rates is None or unpriv_rates is None:
            return float('nan')

        if np.isnan(priv_rates['fpr']) or np.isnan(unpriv_rates['fpr']):
            return float('nan')

        return unpriv_rates['fpr'] - priv_rates['fpr']
    
    def _false_negative_rate_difference(self) -> float:
        """Calculate FNR Difference"""
        priv_rates = self._confusion_matrix_rates(self.privileged_df, self.privileged_pred_df)
        unpriv_rates = self._confusion_matrix_rates(self.unprivileged_df, self.unprivileged_pred_df)
        
        if priv_rates is None or unpriv_rates is None:
            return float('nan')

        if np.isnan(priv_rates['fnr']) or np.isnan(unpriv_rates['fnr']):
            return float('nan')

        return unpriv_rates['fnr'] - priv_rates['fnr']
    
    def _false_discovery_rate_difference(self) -> float:
        """Calculate FDR Difference"""
        priv_rates = self._confusion_matrix_rates(self.privileged_df, self.privileged_pred_df)
        unpriv_rates = self._confusion_matrix_rates(self.unprivileged_df, self.unprivileged_pred_df)
        
        if priv_rates is None or unpriv_rates is None:
            return float('nan')

        if np.isnan(priv_rates['fdr']) or np.isnan(unpriv_rates['fdr']):
            return float('nan')

        return unpriv_rates['fdr'] - priv_rates['fdr']
    
    def _false_omission_rate_difference(self) -> float:
        """Calculate FOR Difference"""
        priv_rates = self._confusion_matrix_rates(self.privileged_df, self.privileged_pred_df)
        unpriv_rates = self._confusion_matrix_rates(self.unprivileged_df, self.unprivileged_pred_df)
        
        if priv_rates is None or unpriv_rates is None:
            return float('nan')

        if np.isnan(priv_rates['for']) or np.isnan(unpriv_rates['for']):
            return float('nan')

        return unpriv_rates['for'] - priv_rates['for']
    
    def _error_rate_difference(self) -> float:
        """Calculate Error Rate Difference"""
        priv_rates = self._confusion_matrix_rates(self.privileged_df, self.privileged_pred_df)
        unpriv_rates = self._confusion_matrix_rates(self.unprivileged_df, self.unprivileged_pred_df)
        
        if priv_rates is None or unpriv_rates is None:
            return float('nan')

        if np.isnan(priv_rates['error_rate']) or np.isnan(unpriv_rates['error_rate']):
            return float('nan')

        return unpriv_rates['error_rate'] - priv_rates['error_rate']
    
    def _theil_index(self) -> float:
        """Calculate Theil Index (Generalized Entropy Index with α=1)"""
        # Guard if predictions are missing
        if self.dataset_pred is None:
            return float('nan')
        # Get predictions
        if self.pred_label_col in self.dataset_pred.columns:
            y_pred = self.dataset_pred[self.pred_label_col].values
        elif self.label_column in self.dataset_pred.columns:
            y_pred = self.dataset_pred[self.label_column].values
        else:
            return float('nan')

        # Calculate benefit (1 for correct prediction, 0 for incorrect)
        y_true = self.dataset[self.label_column].values
        benefits = (y_pred == y_true).astype(float)
        
        # Calculate mean benefit
        mu = benefits.mean() if len(benefits) > 0 else 0

        if mu == 0:
            return 0.0
        
        # Calculate Theil index
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-10
        benefits_safe = np.where(benefits == 0, epsilon, benefits)
        
        theil = np.mean((benefits_safe / mu) * np.log(benefits_safe / mu))
        
        return theil

    # New classification metric implementations
    def _predictive_parity_difference(self) -> float:
        """Predictive Parity Difference: difference in Positive Predictive Value (PPV) between groups."""
        priv_rates = self._confusion_matrix_rates(self.privileged_df, self.privileged_pred_df)
        unpriv_rates = self._confusion_matrix_rates(self.unprivileged_df, self.unprivileged_pred_df)

        if priv_rates is None or unpriv_rates is None:
            return float('nan')

        ppv_priv = priv_rates.get('ppv', np.nan)
        ppv_unpriv = unpriv_rates.get('ppv', np.nan)

        if np.isnan(ppv_priv) or np.isnan(ppv_unpriv):
            return float('nan')

        return ppv_unpriv - ppv_priv

    def _selection_rate_difference(self) -> float:
        """Selection Rate Difference for predictions (same as statistical parity on predictions)."""
        return self._statistical_parity_difference(use_predictions=True)

    def _balanced_accuracy_difference(self) -> float:
        """Balanced Accuracy Difference: (0.5*(TPR+TNR))_unpriv - (0.5*(TPR+TNR))_priv"""
        priv_rates = self._confusion_matrix_rates(self.privileged_df, self.privileged_pred_df)
        unpriv_rates = self._confusion_matrix_rates(self.unprivileged_df, self.unprivileged_pred_df)

        if priv_rates is None or unpriv_rates is None:
            return float('nan')

        tpr_priv = priv_rates.get('tpr', np.nan)
        tnr_priv = priv_rates.get('tnr', np.nan)
        tpr_unpriv = unpriv_rates.get('tpr', np.nan)
        tnr_unpriv = unpriv_rates.get('tnr', np.nan)

        if np.isnan(tpr_priv) or np.isnan(tnr_priv) or np.isnan(tpr_unpriv) or np.isnan(tnr_unpriv):
            return float('nan')

        bal_priv = 0.5 * (tpr_priv + tnr_priv)
        bal_unpriv = 0.5 * (tpr_unpriv + tnr_unpriv)

        return bal_unpriv - bal_priv

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
        
        # Handle NaN/None
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return (None, 'Metric undefined (insufficient data or division by zero).')

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
            is_fair = value >= threshold.get('min', 0)
            if is_fair:
                interpretation = f"Consistency score ({value:.4f}) is good, indicating similar individuals are treated similarly."
            else:
                interpretation = f"Consistency score ({value:.4f}) is low, suggesting individual fairness concerns."
        
        elif metric_key in ['theil_index']:
            # Lower is better (closer to 0)
            is_fair = value <= threshold.get('max', 0.1)
            if is_fair:
                interpretation = f"Theil index ({value:.4f}) is low, indicating good fairness in benefit distribution."
            else:
                interpretation = f"Theil index ({value:.4f}) is high, indicating inequality in outcomes."
        
        else:
            # Difference metrics: should be close to 0
            is_fair = threshold.get('min', -0.1) <= value <= threshold.get('max', 0.1)
            if is_fair:
                interpretation = f"The difference ({value:.4f}) is within acceptable bounds ({threshold.get('min')} to {threshold.get('max')})."
            elif value < threshold.get('min', -0.1):
                interpretation = f"The difference ({value:.4f}) is significantly negative, indicating the unprivileged group is disadvantaged."
            else:
                interpretation = f"The difference ({value:.4f}) is significantly positive, indicating the privileged group is disadvantaged."
        
        return is_fair, interpretation
