"""
Core bias detection calculation engine using AIF360
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric

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
        self.label_mapping = label_mapping

        # Basic validation and preparation
        self._validate_inputs()

        # Alignment logic (inherited from previous implementation)
        self.privileged_df = None # Kept for compatibility if accessed directly, though aif360 handles it
        self.unprivileged_df = None

        # Filter dataset for binary groups for legacy attributes (alignment uses them)
        self.privileged_df = self.dataset[self.dataset[protected_attr] == privileged_value]
        self.unprivileged_df = self.dataset[self.dataset[protected_attr] == unprivileged_value]

        # Prepare prediction DataFrames (aligned to original dataset if provided)
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

        # Prepare AIF360 datasets
        self._prepare_aif360_datasets()

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

    def _prepare_aif360_datasets(self):
        """Convert inputs to AIF360 compatible datasets"""
        # Determine source dataframe for ground truth
        # If we have merged predictions, use that as source to ensure alignment
        if self.detection_type == "Model Bias Detection" and self.dataset_pred is not None:
            source_df = self.dataset_pred.copy()
        else:
            source_df = self.dataset.copy()

        # Filter to only privileged and unprivileged groups
        mask = source_df[self.protected_attr].isin([self.privileged_value, self.unprivileged_value])
        filtered_df = source_df[mask].copy()

        # Map protected attribute to 1 (privileged) and 0 (unprivileged)
        mapping = {self.privileged_value: 1, self.unprivileged_value: 0}
        filtered_df[self.protected_attr] = filtered_df[self.protected_attr].map(mapping)

        # 1. Ground Truth Dataset (aif_dataset)
        self.aif_dataset = BinaryLabelDataset(
            favorable_label=1,
            unfavorable_label=0,
            df=filtered_df,
            label_names=[self.label_column],
            protected_attribute_names=[self.protected_attr]
        )

        # keep a pandas copy of the filtered dataframe for mitigation transforms
        self._filtered_df = filtered_df.copy()

        self.privileged_groups = [{self.protected_attr: 1}]
        self.unprivileged_groups = [{self.protected_attr: 0}]

        # 2. Predictions Dataset (aif_dataset_pred)
        self.aif_dataset_pred = None
        if self.detection_type == "Model Bias Detection" and self.dataset_pred is not None:
            # We create a copy of filtered_df but replace labels with predictions
            df_pred = filtered_df.copy()

            # self.pred_label_col should exist in source_df (which is dataset_pred)
            if self.pred_label_col not in df_pred.columns:
                 raise ValueError(f"Predicted columns {self.pred_label_col} missing after alignment")

            # Set the label column to the predicted values
            df_pred[self.label_column] = df_pred[df_pred.columns[df_pred.columns.get_loc(self.pred_label_col)]]

            self.aif_dataset_pred = BinaryLabelDataset(
                favorable_label=1,
                unfavorable_label=0,
                df=df_pred,
                label_names=[self.label_column],
                protected_attribute_names=[self.protected_attr]
            )

    def calculate_metrics(self, selected_metrics: List[str]) -> Dict:
        """
        Calculate selected fairness metrics using AIF360
        """
        results = {
            'summary': self._calculate_summary(),
            'metrics': {}
        }

        # Initialize AIF360 Metric Objects
        dataset_metric = BinaryLabelDatasetMetric(
            self.aif_dataset,
            unprivileged_groups=self.unprivileged_groups,
            privileged_groups=self.privileged_groups
        )

        classification_metric = None
        if self.aif_dataset_pred is not None:
            classification_metric = ClassificationMetric(
                self.aif_dataset,
                self.aif_dataset_pred,
                unprivileged_groups=self.unprivileged_groups,
                privileged_groups=self.privileged_groups
            )

        for metric_key in selected_metrics:
            notes = []
            reliable = True
            metric_value = None
            try:
                # Check group sizes (still important)
                # aif_dataset.protected_attributes is numpy array, shape (n, 1) or (n,)
                priv_mask = self.aif_dataset.protected_attributes == 1
                unpriv_mask = self.aif_dataset.protected_attributes == 0
                priv_count = np.sum(priv_mask)
                unpriv_count = np.sum(unpriv_mask)

                if priv_count < self.min_group_size or unpriv_count < self.min_group_size:
                    reliable = False
                    notes.append(f"One or both groups have < {self.min_group_size} samples; metric may be unreliable.")

                if self.detection_type == "Dataset Bias Detection":
                     metric_value = self._get_dataset_metric(dataset_metric, metric_key)
                else:
                     if classification_metric is None:
                         raise ValueError("Predictions required for classification metrics")
                     metric_value = self._get_classification_metric(classification_metric, metric_key)

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

        results['summary']['bias_detected'] = any(
            (not m['is_fair']) for m in results['metrics'].values() if isinstance(m.get('is_fair'), bool)
        )

        return results

    def _calculate_summary(self) -> Dict:
        """Calculate summary statistics"""
        total = self.aif_dataset.features.shape[0]
        # In aif_dataset, protected attr is at the index of protected_attribute_names
        # But we can access it via protected_attributes
        priv_count = int(np.sum(self.aif_dataset.protected_attributes == 1))
        unpriv_count = int(np.sum(self.aif_dataset.protected_attributes == 0))

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

    def _get_dataset_metric(self, metric_obj: BinaryLabelDatasetMetric, metric_key: str) -> float:
        """Calculate dataset-level fairness metrics using aif360"""

        if metric_key == 'statistical_parity_difference':
            return metric_obj.mean_difference()

        elif metric_key == 'disparate_impact':
            return metric_obj.disparate_impact()

        elif metric_key == 'consistency':
            # consistency returns an array of values per sample, we average it
            # default n_neighbors=5
            cons = metric_obj.consistency(n_neighbors=5)
            # AIF360 returns one value per sample. The paper definition usually averages them.
            # aif360 implementation: returns array-like (n_samples,)
            return float(np.mean(cons))

        elif metric_key == 'base_rate':
            return metric_obj.base_rate()

        elif metric_key == 'selection_rate':
            # Alias for statistical parity difference on labels
            return metric_obj.mean_difference()

        elif metric_key == 'positive_rate_ratio':
            # Alias for disparate impact (positive rate ratio)
            return metric_obj.disparate_impact()

        else:
            raise ValueError(f"Unknown dataset metric: {metric_key}")

    def mitigate_dataset(self, method: str, **kwargs) -> pd.DataFrame:
        """Apply a dataset-level mitigation and return a pandas DataFrame with the mitigated data.

        Supported methods:
            - 'reweighing' : adds an 'instance_weight' column calculated by AIF360 Reweighing
            - 'disparate_impact_remover' : returns a repaired DataFrame with features adjusted
            - 'optimized_preprocessing' : attempts to run AIF360's optimized preprocessor (best-effort)
        """
        if not hasattr(self, '_filtered_df') or self._filtered_df is None:
            raise ValueError('Filtered dataframe not available for mitigation. Ensure _prepare_aif360_datasets ran successfully.')

        method = method.lower()
        if method == 'reweighing':
            try:
                from aif360.algorithms.preprocessing import Reweighing
            except Exception as e:
                raise ImportError('Reweighing not available in aif360 installation') from e

            rw = Reweighing(unprivileged_groups=self.unprivileged_groups, privileged_groups=self.privileged_groups)
            transformed = rw.fit_transform(self.aif_dataset)

            # transformed.instance_weights aligns with _filtered_df ordering
            weights = getattr(transformed, 'instance_weights', None)
            if weights is None:
                raise RuntimeError('Reweighing did not produce instance weights')

            out_df = self._filtered_df.copy()
            out_df['instance_weight'] = weights
            return out_df

        elif method == 'disparate_impact_remover':
            try:
                from aif360.algorithms.preprocessing import DisparateImpactRemover
            except Exception as e:
                raise ImportError('DisparateImpactRemover not available in aif360 installation') from e

            repair_level = float(kwargs.get('repair_level', 1.0))
            dir_ = DisparateImpactRemover(repair_level=repair_level)
            # DisparateImpactRemover expects either a BinaryLabelDataset or in some versions a DataFrame
            try:
                # Prefer passing the aif_dataset (BinaryLabelDataset) since many implementations expect it
                if hasattr(dir_, 'fit_transform'):
                    repaired = dir_.fit_transform(self.aif_dataset)
                elif hasattr(dir_, 'predict'):
                    repaired = dir_.predict(self.aif_dataset)
                elif hasattr(dir_, 'transform'):
                    repaired = dir_.transform(self.aif_dataset)
                else:
                    raise RuntimeError('DisparateImpactRemover does not expose fit_transform/transform/predict methods in this AIF360 version')
            except Exception as e:
                raise RuntimeError(f'DisparateImpactRemover failed: {e}') from e

            out_df = None
            # If returned as a BinaryLabelDataset, convert to DataFrame
            try:
                if isinstance(repaired, BinaryLabelDataset):
                    out_df = repaired.convert_to_dataframe()[0]
                elif hasattr(repaired, 'convert_to_dataframe'):
                    out_df = repaired.convert_to_dataframe()[0]
                elif isinstance(repaired, pd.DataFrame):
                    out_df = repaired
                elif isinstance(repaired, np.ndarray):
                    out_df = pd.DataFrame(repaired, columns=self._filtered_df.columns, index=self._filtered_df.index)
            except Exception as e:
                raise RuntimeError(f'Failed to convert DisparateImpactRemover output to DataFrame: {e}') from e

            if out_df is None:
                raise RuntimeError('DisparateImpactRemover produced unsupported output format')

            # Ensure label and protected columns preserved if DR only changed features
            if self.label_column not in out_df.columns:
                out_df[self.label_column] = self._filtered_df[self.label_column].values
            if self.protected_attr not in out_df.columns:
                out_df[self.protected_attr] = self._filtered_df[self.protected_attr].values

            return out_df

        elif method == 'optimized_preprocessing':
            # Optimized preprocessor API has changed across AIF360 versions; try best-effort
            try:
                # many aif360 versions expose the class in this path
                from aif360.algorithms.preprocessing import OptimizedPreprocessing
            except Exception:
                try:
                    from aif360.algorithms.preprocessing.optim_preproc import OptimizedPreprocessing
                except Exception as e:
                    raise ImportError('Optimized Preprocessing not available in aif360 installation') from e

            # Provide a conservative, simple configuration for optimized preprocessing
            opt_params = kwargs.get('opt_params', {})
            op = OptimizedPreprocessing(unprivileged_groups=self.unprivileged_groups,
                                        privileged_groups=self.privileged_groups,
                                        **(opt_params or {}))
            try:
                transformed = op.fit_transform(self.aif_dataset)
            except Exception:
                transformed = op.transform(self.aif_dataset)

            # Try to extract a DataFrame from the returned dataset-like object
            if hasattr(transformed, 'convert_to_dataframe'):
                try:
                    out_df = transformed.convert_to_dataframe()[0]
                except Exception:
                    out_df = None
            else:
                out_df = None

            if out_df is None:
                # Fallback: try to construct DataFrame from features/labels/protected_attributes
                try:
                    feats = getattr(transformed, 'features', None)
                    labs = getattr(transformed, 'labels', None)
                    prot = getattr(transformed, 'protected_attributes', None)
                    if feats is not None:
                        cols = self._filtered_df.columns.tolist()
                        out_df = pd.DataFrame(feats, columns=cols, index=self._filtered_df.index)
                        if labs is not None:
                            out_df[self.label_column] = labs
                        if prot is not None:
                            out_df[self.protected_attr] = prot
                    else:
                        raise RuntimeError('OptimizedPreprocessing produced an unsupported output format')
                except Exception as e:
                    raise RuntimeError('Failed to convert OptimizedPreprocessing output to DataFrame') from e

            return out_df

        else:
            raise ValueError(f"Unknown mitigation method: {method}")

    def _get_classification_metric(self, metric_obj: ClassificationMetric, metric_key: str) -> float:
        """Calculate classification-level fairness metrics using aif360"""

        if metric_key == 'equal_opportunity_difference':
            return metric_obj.equal_opportunity_difference()

        elif metric_key == 'average_odds_difference':
            return metric_obj.average_odds_difference()

        elif metric_key == 'statistical_parity_difference':
            # This is selection rate difference on predictions
            # ClassificationMetric inherits statistical_parity_difference from BinaryLabelDatasetMetric
            # which operates on the 'dataset' (which in ClassificationMetric is the predicted dataset? No)
            # Wait, ClassificationMetric init(dataset, classified_dataset).
            # It inherits from BinaryLabelDatasetMetric initialized with 'classified_dataset'.
            # So calling statistical_parity_difference() on ClassificationMetric returns SPD of predictions.
            return metric_obj.statistical_parity_difference()

        elif metric_key == 'disparate_impact':
             return metric_obj.disparate_impact()

        elif metric_key == 'false_positive_rate_difference':
            return metric_obj.false_positive_rate_difference()

        elif metric_key == 'false_negative_rate_difference':
            return metric_obj.false_negative_rate_difference()

        elif metric_key == 'false_discovery_rate_difference':
            return metric_obj.false_discovery_rate_difference()

        elif metric_key == 'false_omission_rate_difference':
            return metric_obj.false_omission_rate_difference()

        elif metric_key == 'error_rate_difference':
            return metric_obj.error_rate_difference()

        elif metric_key == 'theil_index':
            return metric_obj.theil_index()

        elif metric_key == 'predictive_parity_difference':
            # difference in PPV
            # AIF360 doesn't have predictive_parity_difference directly?
            # It has positive_predictive_value(privileged=True) etc.
            ppv_unpriv = metric_obj.positive_predictive_value(privileged=False)
            ppv_priv = metric_obj.positive_predictive_value(privileged=True)
            return ppv_unpriv - ppv_priv

        elif metric_key == 'selection_rate_difference':
            return metric_obj.statistical_parity_difference()

        elif metric_key == 'balanced_accuracy_difference':
            # (tpr+tnr)/2
            # AIF360 has generalized_entropy_index, etc. checking docs for balanced_accuracy.
            # Not directly difference. implement manually.

            def balanced_accuracy(privileged):
                tpr = metric_obj.true_positive_rate(privileged=privileged)
                tnr = metric_obj.true_negative_rate(privileged=privileged)
                return 0.5 * (tpr + tnr)

            return balanced_accuracy(privileged=False) - balanced_accuracy(privileged=True)

        else:
            raise ValueError(f"Unknown classification metric: {metric_key}")

    def _interpret_metric(self, metric_key: str, value: float) -> tuple:
        """
        Interpret metric value to determine if it indicates bias
        """
        from metrics_info import DATASET_METRICS, CLASSIFICATION_METRICS

        if self.detection_type == "Dataset Bias Detection":
            metric_info = DATASET_METRICS.get(metric_key, {})
        else:
            metric_info = CLASSIFICATION_METRICS.get(metric_key, {})

        threshold = metric_info.get('threshold', {})

        if value is None or (isinstance(value, float) and np.isnan(value)):
            return (None, 'Metric undefined (insufficient data or division by zero).')

        if metric_key in ['disparate_impact']:
            is_fair = threshold['min'] <= value <= threshold['max']
            if is_fair:
                interpretation = f"The ratio ({value:.4f}) falls within the acceptable range of {threshold['min']}-{threshold['max']}."
            elif value < threshold['min']:
                interpretation = f"The ratio ({value:.4f}) is below {threshold['min']}, indicating the unprivileged group is disadvantaged."
            else:
                interpretation = f"The ratio ({value:.4f}) is above {threshold['max']}, indicating the privileged group is disadvantaged."

        elif metric_key in ['consistency']:
            is_fair = value >= threshold.get('min', 0)
            if is_fair:
                interpretation = f"Consistency score ({value:.4f}) is good, indicating similar individuals are treated similarly."
            else:
                interpretation = f"Consistency score ({value:.4f}) is low, suggesting individual fairness concerns."

        elif metric_key in ['theil_index']:
            is_fair = value <= threshold.get('max', 0.1)
            if is_fair:
                interpretation = f"Theil index ({value:.4f}) is low, indicating good fairness in benefit distribution."
            else:
                interpretation = f"Theil index ({value:.4f}) is high, indicating inequality in outcomes."

        else:
            is_fair = threshold.get('min', -0.1) <= value <= threshold.get('max', 0.1)
            if is_fair:
                interpretation = f"The difference ({value:.4f}) is within acceptable bounds ({threshold.get('min')} to {threshold.get('max')})."
            elif value < threshold.get('min', -0.1):
                interpretation = f"The difference ({value:.4f}) is significantly negative, indicating the unprivileged group is disadvantaged."
            else:
                interpretation = f"The difference ({value:.4f}) is significantly positive, indicating the privileged group is disadvantaged."

        return is_fair, interpretation
