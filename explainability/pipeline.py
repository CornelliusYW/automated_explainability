import mlflow
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
import shap

from explainability.data_models import ExplainabilityReport
from explainability.explainers import SHAPExplainer, LIMEExplainer
from explainability.llm_utils import generate_llm_summary

class ExplainabilityPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.use_shap = config.get("use_shap", False)
        self.use_lime = config.get("use_lime", False)

    def run_explainability_check(
        self, model: Any, X: pd.DataFrame, y: pd.Series, model_version: str
    ) -> ExplainabilityReport:
        with mlflow.start_run():
            # Global feature importance
            if self.use_shap:
                global_importance = self.compute_shap_global(model, X)
            else:
                global_importance = {col: 0.0 for col in X.columns}

            # Local explanations for a sample of 20 instances
            sample_idx = np.random.choice(len(X), size=min(20, len(X)), replace=False)
            local_explanations = {}
            for idx in sample_idx:
                xi = X.iloc[[idx]]
                if self.use_shap:
                    local_imp = self.compute_shap_local(model, xi)
                elif self.use_lime:
                    local_imp = self.compute_lime_local(model, xi)
                else:
                    local_imp = {col: 0.0 for col in X.columns}
                local_explanations[int(idx)] = local_imp

            # Log visualizations
            if self.use_shap:
                self._log_shap_summary_plot(model, X)
            if self.use_lime:
                self._log_lime_explanation(model, X)

            # Generate LLM summary directly using the raw info
            try:
                summary = generate_llm_summary(
                    global_importance=global_importance,
                    local_explanations=local_explanations,
                    model_version=model_version
                )
            except Exception as e:
                logging.error(f"LLM summary generation failed: {e}")
                summary = "LLM summary generation failed."


            # Build the final report
            report = ExplainabilityReport(
                timestamp=datetime.now(),
                model_version=model_version,
                global_feature_importance=global_importance,
                local_explanations=local_explanations,
                summary_explanation=summary
            )

            # Log the report to MLflow
            self._log_to_mlflow(report)

            return report


    def compute_shap_global(self, model, X: pd.DataFrame) -> Dict[str, float]:
        explainer = SHAPExplainer()
        global_exp = explainer.explain_global(model, X)
        
        feature_importance = global_exp.get("feature_importance", {})
        result = {}
        for feature, val in feature_importance.items():
            if isinstance(val, np.ndarray):
                # If it's an array, take mean of absolute values
                result[feature] = float(np.mean(np.abs(val)))
            else:
                result[feature] = float(abs(val))
        return result


    def compute_shap_local(self, model: Any, xi: pd.DataFrame) -> Dict[str, float]:
        explainer = SHAPExplainer()
        local_exp = explainer.explain_local(model, xi, instances=[0])
        shap_vals = local_exp.get("explanations", {}).get(0, {}).get("shap_values", {})

        result = {}
        for feature, val in shap_vals.items():
            if isinstance(val, np.ndarray):
                # Take mean absolute if multiple, else extract the single value
                if val.size == 1:
                    result[feature] = float(abs(val.item()))
                else:
                    result[feature] = float(np.mean(np.abs(val)))
            else:
                result[feature] = float(abs(val))
        return result if result else {col: 0.0 for col in xi.columns}


    def compute_lime_local(self, model: Any, xi: pd.DataFrame) -> Dict[str, float]:
        explainer = LIMEExplainer()
        local_exp = explainer.explain_local(model, xi, instances=[0])
        lime_vals = local_exp.get("explanations", {}).get(0, {}).get("lime_explanation", {})
        return {feature: float(abs(val)) for feature, val in lime_vals.items()} if lime_vals else {col: 0.0 for col in xi.columns}

    def _log_to_mlflow(self, report: ExplainabilityReport) -> None:
        report_json = report.to_json()
        filename = f"explainability_report_{report.model_version}_{report.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w") as f:
            f.write(report_json)
        mlflow.log_artifact(filename, artifact_path="explainability_reports")

    def _log_shap_summary_plot(self, model: Any, X: pd.DataFrame) -> None:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X.sample(min(500, len(X)), random_state=42))
            
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X.sample(min(500, len(X)), random_state=42), plot_type="bar", max_display=10, show=False)
            plt.title("Top 10 Global Feature Importances")
            plt.tight_layout()
            plt.savefig("shap_summary_plot.png", bbox_inches="tight")
            plt.close()
            mlflow.log_artifact("shap_summary_plot.png", artifact_path="explainability_plots")
        except Exception as e:
            logging.error(f"Failed to generate SHAP summary plot: {e}")


    def _log_lime_explanation(self, model: Any, X: pd.DataFrame) -> None:
        try:
            explainer = LimeTabularExplainer(
                X.values,
                feature_names=X.columns.tolist(),
                mode="classification",
                discretize_continuous=True,
            )
            exp = explainer.explain_instance(
                X.iloc[0].values,
                model.predict_proba if hasattr(model, "predict_proba") else model.predict,
                num_features=min(10, X.shape[1]),
            )
            # Save the image
            exp.save_to_file("lime_explanation.html")
            mlflow.log_artifact("lime_explanation.html", artifact_path="explainability_plots")

        except Exception as e:
            logging.error(f"Failed to generate LIME explanation: {e}")

