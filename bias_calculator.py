"""
Core bias detection calculation engine using AIF360
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from metrics_info import DATASET_METRICS, CLASSIFICATION_METRICS

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

    def _validate_inputs(self):
        """Validate inputs before processing"""
        if self.protected_attr not in self.dataset.columns:
            raise ValueError(f"Protected attribute '{self.protected_attr}' not found in dataset")
        if self.label_column not in self.dataset.columns:
            raise ValueError(f"Label column '{self.label_column}' not found in dataset")

        # Convert privileged/unprivileged values to match the dtype of the protected attribute column
        # This handles cases where values come as strings from web forms but dataset has integers
        col_dtype = self.dataset[self.protected_attr].dtype
        try:
            if col_dtype in ['int64', 'int32', 'int16', 'int8']:
                self.privileged_value = int(self.privileged_value)
                self.unprivileged_value = int(self.unprivileged_value)
            elif col_dtype in ['float64', 'float32', 'float16']:
                self.privileged_value = float(self.privileged_value)
                self.unprivileged_value = float(self.unprivileged_value)
            # For object/string types, keep as-is or convert to string
            elif col_dtype == 'object':
                # Try to match the type of existing values
                sample_val = self.dataset[self.protected_attr].dropna().iloc[0] if len(self.dataset[self.protected_attr].dropna()) > 0 else None
                if sample_val is not None:
                    if isinstance(sample_val, (int, np.integer)):
                        self.privileged_value = int(self.privileged_value)
                        self.unprivileged_value = int(self.unprivileged_value)
                    elif isinstance(sample_val, (float, np.floating)):
                        self.privileged_value = float(self.privileged_value)
                        self.unprivileged_value = float(self.unprivileged_value)
                    else:
                        self.privileged_value = str(self.privileged_value)
                        self.unprivileged_value = str(self.unprivileged_value)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Could not convert privileged/unprivileged values to match column dtype {col_dtype}: {e}")

        # Check that privileged and unprivileged groups exist
        if self.privileged_value not in self.dataset[self.protected_attr].unique():
            raise ValueError(f"Privileged value {self.privileged_value} (type: {type(self.privileged_value)}) not found in protected attribute column. Available values: {self.dataset[self.protected_attr].unique()}")
        if self.unprivileged_value not in self.dataset[self.protected_attr].unique():
            raise ValueError(f"Unprivileged value {self.unprivileged_value} (type: {type(self.unprivileged_value)}) not found in protected attribute column. Available values: {self.dataset[self.protected_attr].unique()}")
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

        # Keep an unmapped copy so we can restore original protected-attribute values
        # when returning mitigated DataFrames (prevents losing original category labels)
        self._filtered_original = filtered_df.copy()

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

        # If the raw dataframe contained instance weights (from Reweighing), set them on the aif dataset
        for wname in ('instance_weight', 'instance_weights', 'sample_weight'):
            if wname in filtered_df.columns:
                try:
                    weights = filtered_df[wname].values
                    # aif360 expects a 2D array for instance_weights
                    self.aif_dataset.instance_weights = weights.reshape(-1, 1)
                except Exception:
                    # ignore if cannot set
                    pass

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

    def _get_classification_metric(self, metric_obj: ClassificationMetric, metric_key: str) -> float:
        """Calculate classification-level fairness metrics using aif360 ClassificationMetric"""
        # Map known metric keys to ClassificationMetric methods
        try:
            if metric_key == 'statistical_parity_difference':
                return metric_obj.statistical_parity_difference()
            elif metric_key == 'disparate_impact':
                return metric_obj.disparate_impact()
            elif metric_key == 'equal_opportunity_difference':
                return metric_obj.equal_opportunity_difference()
            elif metric_key == 'average_odds_difference':
                return metric_obj.average_odds_difference()
            elif metric_key == 'false_positive_rate_difference':
                return metric_obj.false_positive_rate_difference()
            elif metric_key == 'false_negative_rate_difference':
                return metric_obj.false_negative_rate_difference()
            elif metric_key == 'theil_index':
                return metric_obj.theil_index()
            elif metric_key == 'predictive_parity_difference':
                # Not always implemented; attempt to compute via precision per group if available
                return metric_obj.positive_predictive_value_difference()
            else:
                # Fallback: try to call a method with same name on the metric object
                if hasattr(metric_obj, metric_key):
                    func = getattr(metric_obj, metric_key)
                    return func()
                raise ValueError(f"Unknown classification metric: {metric_key}")
        except Exception as e:
            # Surface a helpful error
            raise RuntimeError(f"Error computing classification metric '{metric_key}': {e}") from e

    def _interpret_metric(self, metric_key: str, value: Any):
        """Interpret metric value against thresholds from metrics_info and return (is_fair, interpretation_str)."""
        try:
            if value is None:
                return (None, 'Metric value not available')
            if isinstance(value, (list, tuple, np.ndarray)):
                # reduce to a scalar when possible
                try:
                    value = float(np.mean(value))
                except Exception:
                    return (None, 'Metric returned a non-scalar value')

            val = float(value)

            # Determine metric info source
            if metric_key in DATASET_METRICS:
                info = DATASET_METRICS.get(metric_key, {})
            else:
                info = CLASSIFICATION_METRICS.get(metric_key, {})

            thr = info.get('threshold', {})
            min_thr = thr.get('min', None)
            max_thr = thr.get('max', None)

            # If thresholds are available, use them
            if min_thr is not None and max_thr is not None:
                is_fair = (val >= min_thr) and (val <= max_thr)
                interpretation = f"Value {val:.4f}; acceptable range [{min_thr}, {max_thr}]"
                if not is_fair:
                    interpretation += ". Outside acceptable bounds."
                return (bool(is_fair), interpretation)

            # Fallback heuristics
            if 'disparate_impact' in metric_key or metric_key == 'positive_rate_ratio':
                is_fair = (0.8 <= val <= 1.25)
                interpretation = f"Disparate impact ratio {val:.4f}; 0.8-1.25 rule applied."
                return (is_fair, interpretation)

            # Default for difference metrics: small absolute difference is fair
            is_fair = abs(val) <= 0.1
            interpretation = f"Difference {val:.4f}; threshold ±0.1 applied."
            return (is_fair, interpretation)
        except Exception as e:
            return (None, f'Error interpreting metric: {e}')

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

            # transformed may expose instance weights under different names and shapes
            weights = None
            for attr in ('instance_weights', 'instance_weight', 'instanceWeights'):
                weights = getattr(transformed, attr, None)
                if weights is not None:
                    break

            if weights is None:
                # Some older/newer versions place weights as attribute on the returned object
                raise RuntimeError('Reweighing did not produce instance weights')

            # Flatten if it's 2D
            try:
                weights_arr = np.array(weights).reshape(-1)
            except Exception:
                weights_arr = np.array(weights).ravel()

            out_df = self._filtered_df.copy()
            out_df['instance_weight'] = weights_arr
            # Restore original protected attribute values before returning
            try:
                if hasattr(self, '_filtered_original') and self.protected_attr in self._filtered_original.columns:
                    out_df[self.protected_attr] = self._filtered_original[self.protected_attr].values
            except Exception:
                pass
            return out_df

        elif method == 'disparate_impact_remover':
            try:
                from aif360.algorithms.preprocessing import DisparateImpactRemover
            except Exception as e:
                raise ImportError('DisparateImpactRemover not available in aif360 installation') from e

            repair_level = float(kwargs.get('repair_level', 1.0))

            # Create a temporary DataFrame that keeps the protected attribute as a regular feature
            temp_df = self._filtered_df.copy()
            # Use a duplicate column name to ensure the attribute is present among features
            dup_col = f"{self.protected_attr}_as_feature"
            # If the duplicate name collides (very unlikely), pick a unique suffix
            if dup_col in temp_df.columns:
                i = 1
                while f"{dup_col}_{i}" in temp_df.columns:
                    i += 1
                dup_col = f"{dup_col}_{i}"

            temp_df[dup_col] = temp_df[self.protected_attr]

            # Build a BinaryLabelDataset using the duplicate as the protected attribute
            # so that the DisparateImpactRemover can find it among the features if it
            # expects the protected attribute to be available in the feature list.
            temp_aif = BinaryLabelDataset(
                favorable_label=1,
                unfavorable_label=0,
                df=temp_df,
                label_names=[self.label_column],
                protected_attribute_names=[dup_col]
            )

            dir_ = DisparateImpactRemover(repair_level=repair_level)

            # Some AIF360 versions expose fit_transform, some predict/transform. Try them in order.
            try:
                if hasattr(dir_, 'fit_transform'):
                    repaired = dir_.fit_transform(temp_aif)
                elif hasattr(dir_, 'predict'):
                    repaired = dir_.predict(temp_aif)
                elif hasattr(dir_, 'transform'):
                    repaired = dir_.transform(temp_aif)
                else:
                    raise RuntimeError('DisparateImpactRemover does not expose fit_transform/transform/predict methods in this AIF360 version')
            except Exception as e:
                raise RuntimeError(f'DisparateImpactRemover failed: {e}') from e

            out_df = None
            try:
                # aif360 BinaryLabelDataset exposes convert_to_dataframe() returning (df, meta)
                if isinstance(repaired, BinaryLabelDataset):
                    out_df = repaired.convert_to_dataframe()[0]
                elif hasattr(repaired, 'convert_to_dataframe'):
                    out_df = repaired.convert_to_dataframe()[0]
                elif isinstance(repaired, pd.DataFrame):
                    out_df = repaired
                elif isinstance(repaired, np.ndarray):
                    cols = temp_df.columns.tolist()
                    out_df = pd.DataFrame(repaired, columns=cols, index=temp_df.index)
            except Exception as e:
                raise RuntimeError(f'Failed to convert DisparateImpactRemover output to DataFrame: {e}') from e

            if out_df is None:
                raise RuntimeError('DisparateImpactRemover produced unsupported output format')

            # After repair, ensure the original protected attribute exists and is correct.
            # If DisparateImpactRemover returned the duplicate column, map it back.
            try:
                if self.protected_attr not in out_df.columns and dup_col in out_df.columns:
                    # Rename duplicate back to original protected attribute name
                    out_df[self.protected_attr] = out_df[dup_col]
                elif self.protected_attr in out_df.columns:
                    # keep existing
                    pass
                else:
                    # As a fallback, restore original values from the pre-mitigation df
                    out_df[self.protected_attr] = self._filtered_df[self.protected_attr].values
            except Exception:
                # On any error, ensure original values are present
                out_df[self.protected_attr] = self._filtered_df[self.protected_attr].values

            # If duplicate feature present, drop it to avoid duplicates
            if dup_col in out_df.columns:
                try:
                    out_df.drop(columns=[dup_col], inplace=True)
                except Exception:
                    pass

            # Ensure label column present and properly typed
            if self.label_column in out_df.columns:
                try:
                    out_df[self.label_column] = out_df[self.label_column].astype(int)
                except Exception:
                    # If conversion fails, keep as-is
                    pass

            # Restore original protected attribute values before returning
            try:
                if hasattr(self, '_filtered_original') and self.protected_attr in self._filtered_original.columns:
                    out_df[self.protected_attr] = self._filtered_original[self.protected_attr].values
            except Exception:
                pass

            return out_df

        elif method == 'optimized_preprocessing':
            # Use a simple heuristic-based Optimized Preprocessing if AIF360's implementation is not available
            # This function uses the heuristic provided by the user (flip some negative labels from disadvantaged group)
            def simple_optimized_preprocessing(df, label_col, protected_col, target_parity=0.0, random_state=None):
                df = df.copy()
                if random_state is not None:
                    # shuffle deterministically but avoid bringing old index in as a column
                    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

                groups = df[protected_col].unique()
                if len(groups) != 2:
                    # fallback: return original
                    return df
                g0, g1 = groups

                rate0 = df[df[protected_col] == g0][label_col].mean()
                rate1 = df[df[protected_col] == g1][label_col].mean()

                if rate0 > rate1:
                    disadvantaged = g1
                else:
                    disadvantaged = g0

                # Find negative samples to flip
                candidates = df[(df[protected_col] == disadvantaged) & (df[label_col] == 0)].copy()

                # Number to flip
                gap = abs(rate0 - rate1)
                flips = int(gap * len(df))

                if len(candidates) == 0 or flips <= 0:
                    return df

                flip_idx = candidates.sample(min(flips, len(candidates)), random_state=random_state).index
                df.loc[flip_idx, label_col] = 1

                return df

            target_parity = float(kwargs.get('target_parity', 0.0))
            random_state = kwargs.get('random_state', None)
            out_df = simple_optimized_preprocessing(self._filtered_df.copy(), self.label_column, self.protected_attr, target_parity=target_parity, random_state=random_state)
            # restore original protected attribute values
            try:
                if hasattr(self, '_filtered_original') and self.protected_attr in self._filtered_original.columns:
                    out_df[self.protected_attr] = self._filtered_original[self.protected_attr].values
            except Exception:
                pass
            return out_df
