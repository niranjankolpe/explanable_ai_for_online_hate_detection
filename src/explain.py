"""
explain.py
LIME and SHAP explanations for any model's predict_proba function.
"""

import shap
from lime.lime_text import LimeTextExplainer


def explain_with_lime(text: str, predict_fn, class_names: list, num_features: int = 6) -> dict:
    explainer   = LimeTextExplainer(class_names=class_names)
    explanation = explainer.explain_instance(text, predict_fn, num_features=num_features)
    return dict(explanation.as_list())


def explain_with_shap(text: str, predict_fn, num_features: int = 6) -> dict:
    explainer   = shap.Explainer(predict_fn, masker=shap.maskers.Text(r"\W+"))
    shap_values = explainer([text])
    tokens      = shap_values.data[0]
    values      = shap_values.values[0][:, 1]
    top         = sorted(zip(tokens, values), key=lambda x: abs(x[1]), reverse=True)[:num_features]
    return {token.strip(): float(score) for token, score in top}


def explain_prediction(text: str, predict_fn, class_names: list) -> dict:
    return {
        "lime": explain_with_lime(text, predict_fn, class_names),
        "shap": explain_with_shap(text, predict_fn),
    }
