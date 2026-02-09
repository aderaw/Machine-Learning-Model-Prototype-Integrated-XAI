# ===========================
# Core & backend configuration
# ===========================
import matplotlib
matplotlib.use("Agg")

from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import joblib
import uuid

from sklearn.base import BaseEstimator, TransformerMixin

# =====================================================
# Custom class (MUST match the training-time definition)
# =====================================================
class MLPFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, mlp_model):
        self.mlp_model = mlp_model

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        return self.mlp_model.predict_proba(X_array)

# ===========================
# Load trained hybrid pipeline
# ===========================
pipeline = joblib.load("hybrid_model_pipeline.pkl")

mlp_extractor = pipeline["mlp_extractor"]
xgb = pipeline["xgb_classifier"]
features = pipeline["feature_names"]
class_names = pipeline["class_names"]
X_train_raw = pipeline["X_train_raw"]

# ===========================
# Flask application
# ===========================
app = Flask(__name__, template_folder="templates", static_folder="static")

# ===========================
# Categorical feature mappings
# ===========================
categorical_options = {
    "ClientType": [("E", 0), ("G", 1), ("I", 2)],
    "SEX": [("F", 0), ("M", 1), ("U", 2)],
    "LiteracyLevel": [("U", 0), ("N", 1), ("C", 2), ("K", 3), ("P", 4),
                      ("S", 5), ("DI", 6), ("DG", 7), ("MA", 8), ("MD", 9), ("PHD", 10)],
    "Occupation": [("B", 0), ("E", 1), ("F", 2), ("N", 3), ("O", 4), ("U", 5)],
    "ClientArea": [("R", 0), ("U", 1)],
    "RepaymentFrequency": [("M", 0), ("Q", 1), ("H", 2), ("Y", 3), ("O", 4)]
}

xai_methods = ["SHAP", "LIME"]

# ===========================
# Routes
# ===========================
@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None
    shap_plot_path = None
    lime_plot_path = None
    selected_method = None

    if request.method == "POST":
        raw_input = []
        encoded_input = []

        for feature in features:
            val = request.form.get(feature)
            raw_input.append(val)

            if feature in categorical_options:
                mapping = dict(categorical_options[feature])
                encoded_input.append(mapping.get(val, 0))
            else:
                encoded_input.append(float(val))

        selected_method = request.form.get("method")

        X_input = np.array(encoded_input).reshape(1, -1)
        X_mlp = mlp_extractor.transform(X_input)
        X_combined = np.hstack([X_input, X_mlp])

        pred_idx = xgb.predict(X_combined)[0]
        prediction = class_names[pred_idx]

        # -------- SHAP --------
        if selected_method == "SHAP":
            explainer = shap.TreeExplainer(xgb)
            shap_values = explainer.shap_values(X_combined)

            shap_plot_path = f"static/shap_{uuid.uuid4().hex}.png"

            if isinstance(shap_values, list):
                values = shap_values[pred_idx][0]
                base_value = explainer.expected_value[pred_idx]
            else:
                values = shap_values[0]
                base_value = explainer.expected_value

            expl = shap.Explanation(
                values=values[:len(features)],
                base_values=base_value,
                data=encoded_input,
                feature_names=features
            )

            shap.plots.waterfall(expl, show=False)
            plt.tight_layout()
            plt.savefig(shap_plot_path, bbox_inches="tight")
            plt.close()

        # -------- LIME --------
        elif selected_method == "LIME":
            cat_idx = [i for i, f in enumerate(features) if f in categorical_options]
            cat_names = {i: [k for k, _ in categorical_options[features[i]]] for i in cat_idx}

            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=X_train_raw,
                feature_names=features,
                class_names=class_names,
                categorical_features=cat_idx,
                categorical_names=cat_names,
                mode="classification",
                discretize_continuous=False
            )

            def predict_fn(x):
                X_mlp = mlp_extractor.transform(x)
                return xgb.predict_proba(np.hstack([x, X_mlp]))

            exp = explainer.explain_instance(
                X_input.flatten(),
                predict_fn,
                num_features=len(features)
            )

            lime_plot_path = f"static/lime_{uuid.uuid4().hex}.png"
            fig = exp.as_pyplot_figure()
            fig.savefig(lime_plot_path, bbox_inches="tight")
            plt.close(fig)

        return render_template(
            "result.html",
            prediction=prediction,
            shap_plot_path=shap_plot_path,
            lime_plot_path=lime_plot_path,
            method=selected_method
        )

    return render_template(
        "index.html",
        features=features,
        categorical_options=categorical_options,
        xai_methods=xai_methods
    )

# ===========================
# Local run (Render ignores this)
# ===========================
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
