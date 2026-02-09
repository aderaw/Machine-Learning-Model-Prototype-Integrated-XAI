from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import shap
import lime
import lime.lime_tabular
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neural_network import MLPClassifier
import joblib
import uuid

# ---------------------------
# MLP Feature Extractor Class (No Scaling)
# ---------------------------
class MLPFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, hidden_layer_sizes=(128,), random_state=42):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.random_state = random_state
        self.mlp = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.2
        )
    
    def fit(self, X, y):
        self.mlp.fit(X, y)
        self.coefs_ = self.mlp.coefs_
        self.intercepts_ = self.mlp.intercepts_
        return self

    def transform(self, X):
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        X_hidden = X_array @ self.coefs_[0] + self.intercepts_[0]
        X_hidden = np.maximum(X_hidden, 0)  # ReLU
        return X_hidden

# ---------------------------
# Load saved hybrid pipeline
# ---------------------------
pipeline = joblib.load('hybrid_model_pipeline.pkl')
mlp_extractor = pipeline['mlp_extractor']
xgb = pipeline['xgb_classifier']
features = pipeline['feature_names']
class_names = pipeline['class_names']
X_train_raw = pipeline['X_train_raw']  # RAW features for LIME

# ---------------------------
# Flask app
# ---------------------------
app = Flask(__name__, template_folder='templates', static_folder='static')

# ---------------------------
# Categorical feature mappings
# ---------------------------
categorical_options = {
    'ClientType': [('E', 0), ('G', 1), ('I', 2)],
    'SEX': [('F', 0), ('M', 1), ('U', 2)],
    'LiteracyLevel': [('U', 0), ('N', 1), ('C', 2), ('K', 3), ('P', 4),
                      ('S', 5), ('DI', 6), ('DG', 7), ('MA', 8), ('MD', 9), ('PHD', 10)],
    'Occupation': [('B', 0), ('E', 1), ('F', 2), ('N', 3), ('O', 4), ('U', 5)],
    'ClientArea': [('R', 0), ('U', 1)],
    'BusinessLine': [(str(k), v) for k, v in zip([11,21,31,41,51,61,71,81,82,83,84,85,86,91,92,101], range(16))],
    'LoanPurpose': [(str(k), v) for k, v in zip(range(1, 29), range(28))],
    'RepaymentFrequency': [('M', 0), ('Q', 1), ('H', 2), ('Y', 3), ('O', 4)],
    'ProductID': [(v, i) for i, v in enumerate(['AL','BCL','BMGL','BPSL','BTL','ELOI','ELQ','ELY','HL','IL','ILQ','IPL',
                                                'IYL','IYLH','IYLQ','MEGL','MGT','MIM','MSEGQ','MSEGY','MSEIQ','MSEIS',
                                                'REI','RET','RGT','RYI','RYIH','RYIQ','RYT','SCL','SD','SIL','SL','TL',
                                                'YLHY','YLI','YLT','YNSD'])]
}

xai_methods = ['SHAP', 'LIME']

# ---------------------------
# Routes
# ---------------------------
@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    lime_plot_path = None
    shap_plot_path = None
    selected_method = None

    if request.method == 'POST':
        raw_input_data = []
        input_data = []

        # Encode categorical values to numbers
        for feature in features:
            val = request.form.get(feature)
            raw_input_data.append(val)
            if feature in categorical_options:
                mapping = dict(categorical_options[feature])
                input_data.append(float(mapping.get(val, 0)))
            else:
                input_data.append(float(val))

        selected_method = request.form.get('method')

        # Prepare data for prediction
        X_input = np.array(input_data).reshape(1, -1)
        X_input_mlp = mlp_extractor.transform(X_input)
        X_combined = np.hstack([X_input, X_input_mlp])

        # Prediction
        pred_class_idx = xgb.predict(X_combined)[0]
        prediction = class_names[pred_class_idx]

        # ---------------------------
        # SHAP Explanation
        # ---------------------------
        if selected_method == 'SHAP':
            explainer = shap.TreeExplainer(xgb)
            shap_values = explainer.shap_values(X_combined)
            shap_plot_path = f'static/shap_{uuid.uuid4().hex}.png'

            # Extract correct SHAP values
            if isinstance(shap_values, list):
                shap_vals_for_instance = shap_values[pred_class_idx][0]
                base_value = explainer.expected_value[pred_class_idx]
            else:
                shap_vals_for_instance = shap_values[0, :, pred_class_idx]
                base_value = explainer.expected_value[pred_class_idx]

            # Decode categorical features
            cat_mappings = {f: {v: k for k, v in categorical_options[f]} for f in categorical_options}
            decoded_instance = {}
            for i, col in enumerate(features):
                if col in cat_mappings:
                    decoded_instance[col] = cat_mappings[col].get(int(input_data[i]), raw_input_data[i])
                else:
                    decoded_instance[col] = input_data[i]

            # Only original features for plotting
            feature_names_to_plot = features
            shap_values_to_plot = shap_vals_for_instance[:len(features)]
            data_to_plot = [decoded_instance[f] for f in feature_names_to_plot]

            expl = shap.Explanation(
                values=shap_values_to_plot,
                base_values=base_value,
                data=data_to_plot,
                feature_names=feature_names_to_plot
            )

            shap.plots.waterfall(expl, show=False)
            plt.tight_layout()
            plt.savefig(shap_plot_path, bbox_inches='tight')
            plt.close()

        # ---------------------------
        # LIME Explanation
        # ---------------------------
        elif selected_method == 'LIME':
            categorical_features_idx = [i for i, f in enumerate(features) if f in categorical_options]
            categorical_names = {i: [name for name, val in categorical_options[features[i]]] for i in categorical_features_idx}

            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=X_train_raw,
                feature_names=features,
                class_names=class_names,
                mode='classification',
                categorical_features=categorical_features_idx,
                categorical_names=categorical_names,
                discretize_continuous=False
            )

            def xgb_predict_func(x_array):
                X_mlp = mlp_extractor.transform(x_array)
                X_combined = np.hstack([x_array, X_mlp])
                return xgb.predict_proba(X_combined)

            exp = explainer.explain_instance(
                X_input.flatten(),
                xgb_predict_func,
                num_features=len(features)
            )

            lime_plot_path = f'static/lime_{uuid.uuid4().hex}.png'

            # Extract weights for plotting
            pred_idx = pred_class_idx
            local_pairs = exp.local_exp.get(pred_idx, exp.local_exp[list(exp.local_exp.keys())[0]])
            names = [features[idx] for idx, w in local_pairs]
            weights = np.array([w for idx, w in local_pairs], dtype=float)
            order = np.argsort(np.abs(weights))[::-1]
            names = [names[i] for i in order]
            weights = weights[order]

            top_k = min(21, len(names))
            names = names[:top_k]
            weights = weights[:top_k]

            # Plot LIME
            plt.figure(figsize=(8, 6))
            y_pos = np.arange(len(names))
            colors = ['green' if w > 0 else 'red' for w in weights]
            plt.barh(y_pos, weights, align='center', color=colors)
            plt.yticks(y_pos, names)
            plt.axvline(x=0, color='black', linestyle='--')
            plt.title(f'Local explanation for class {class_names[pred_idx]}')
            plt.grid(True, axis='x', linestyle='--', linewidth=0.5)
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(lime_plot_path, bbox_inches='tight')
            plt.close()

        return render_template('result.html',
                               prediction=prediction,
                               shap_plot_path=shap_plot_path,
                               lime_plot_path=lime_plot_path,
                               method=selected_method)

    return render_template('index.html',
                           features=features,
                           categorical_options=categorical_options,
                           xai_methods=xai_methods)

# ---------------------------
# Run Flask
# ---------------------------
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

