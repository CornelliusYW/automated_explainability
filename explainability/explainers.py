import numpy as np
import shap
from lime.lime_tabular import LimeTabularExplainer
import pandas as pd
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List


class BaseExplainer(ABC):
    @abstractmethod
    def explain_global(self, model, X: pd.DataFrame) -> Dict[str, Any]:
        pass

    @abstractmethod
    def explain_local(self, model, X: pd.DataFrame, instances: List[int]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_feature_importance(self, model, X: pd.DataFrame) -> Dict[str, float]:
        pass


class SHAPExplainer(BaseExplainer):
    def __init__(self):
        self.explainer = None

    def _initialize_explainer(self, model, X_background):
        try:
            if hasattr(model, 'predict_proba'):
                if hasattr(model, 'feature_importances_'):
                    self.explainer = shap.TreeExplainer(model)
                else:
                    self.explainer = shap.KernelExplainer(
                        model.predict_proba, X_background.sample(min(100, len(X_background)))
                    )
            else:
                self.explainer = shap.KernelExplainer(
                    model.predict, X_background.sample(min(100, len(X_background)))
                )
        except Exception as e:
            logging.warning(f"SHAP initialization failed: {e}")
            self.explainer = None

    def explain_global(self, model, X: pd.DataFrame) -> Dict[str, Any]:
        if self.explainer is None:
            self._initialize_explainer(model, X)
        try:
            shap_values = self.explainer.shap_values(X.sample(min(500, len(X))))
            if isinstance(shap_values, list):
                mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
            else:
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
            feature_importance = dict(zip(X.columns, mean_abs_shap))
            return {'method': 'SHAP', 'feature_importance': feature_importance}
        except Exception as e:
            logging.error(f"SHAP global explanation failed: {e}")
            return {'method': 'SHAP', 'error': str(e)}

    def explain_local(self, model, X: pd.DataFrame, instances: List[int]) -> Dict[str, Any]:
        if self.explainer is None:
            self._initialize_explainer(model, X)
        try:
            local_explanations = {}
            X_instances = X.iloc[instances]
            shap_values = self.explainer.shap_values(X_instances)
            
            # Handle both binary (list of 2) and multiclass/multitask outputs
            if isinstance(shap_values, list):
                # If binary classification, typically shap_values[1] is for positive class
                if len(shap_values) == 2:
                    local_vals = shap_values[1]
                else:
                    # Multiclass case, take mean of absolute SHAP values
                    local_vals = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            else:
                local_vals = shap_values

            for i, idx in enumerate(instances):
                instance_shap = dict(zip(X.columns, np.abs(local_vals[i])))
                local_explanations[idx] = {
                    'shap_values': instance_shap,
                    'feature_values': X_instances.iloc[i].to_dict(),
                    'prediction': int(model.predict(X_instances.iloc[[i]])[0])
                }
            return {'method': 'SHAP', 'explanations': local_explanations}
        except Exception as e:
            logging.error(f"SHAP local explanation failed: {e}")
            return {'method': 'SHAP', 'error': str(e)}

    def get_feature_importance(self, model, X: pd.DataFrame) -> Dict[str, float]:
        global_exp = self.explain_global(model, X)
        return global_exp.get('feature_importance', {})


class LIMEExplainer(BaseExplainer):
    def __init__(self, mode='classification'):
        self.mode = mode
        self.explainer = None

    def _initialize_explainer(self, X_train):
        try:
            self.explainer = LimeTabularExplainer(
                X_train.values,
                feature_names=X_train.columns,
                mode=self.mode,
                discretize_continuous=True
            )
        except Exception as e:
            logging.error(f"LIME initialization failed: {e}")
            self.explainer = None

    def explain_global(self, model, X: pd.DataFrame) -> Dict[str, Any]:
        if self.explainer is None:
            self._initialize_explainer(X)
        try:
            sample_size = min(100, len(X))
            sample_indices = np.random.choice(len(X), sample_size, replace=False)
            feature_importance = {col: 0.0 for col in X.columns}
            for idx in sample_indices:
                exp = self.explainer.explain_instance(
                    X.iloc[idx].values,
                    model.predict_proba if hasattr(model, 'predict_proba') else model.predict,
                    num_features=len(X.columns)
                )
                for feature_idx, importance in exp.as_map()[1]:
                    feature_name = X.columns[feature_idx]
                    feature_importance[feature_name] += abs(importance)
            for key in feature_importance:
                feature_importance[key] /= sample_size
            return {'method': 'LIME', 'feature_importance': feature_importance}
        except Exception as e:
            logging.error(f"LIME global explanation failed: {e}")
            return {'method': 'LIME', 'error': str(e)}

    def explain_local(self, model, X: pd.DataFrame, instances: List[int]) -> Dict[str, Any]:
        if self.explainer is None:
            self._initialize_explainer(X)
        try:
            local_explanations = {}
            for idx in instances:
                exp = self.explainer.explain_instance(
                    X.iloc[idx].values,
                    model.predict_proba if hasattr(model, 'predict_proba') else model.predict,
                    num_features=len(X.columns)
                )
                lime_exp_dict = dict(exp.as_list())
                full_explanation = {feat: lime_exp_dict.get(feat, 0.0) for feat in X.columns}
                local_explanations[idx] = {
                    'lime_explanation': full_explanation,
                    'feature_values': X.iloc[idx].to_dict(),
                    'prediction': model.predict(X.iloc[[idx]])[0]
                }
            return {'method': 'LIME', 'explanations': local_explanations}
        except Exception as e:
            logging.error(f"LIME local explanation failed: {e}")
            return {'method': 'LIME', 'error': str(e)}

    def get_feature_importance(self, model, X: pd.DataFrame) -> Dict[str, float]:
        global_exp = self.explain_global(model, X)
        return global_exp.get('feature_importance', {})
