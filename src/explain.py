"""
explain.py
LIME and SHAP explanations for any model's predict_proba function.
"""

import shap
from lime.lime_text import LimeTextExplainer


def create_lime_explainer(class_names: list) -> LimeTextExplainer:
    return LimeTextExplainer(class_names=class_names)


def create_shap_explainer(predict_fn) -> shap.Explainer:
    return shap.Explainer(predict_fn, masker=shap.maskers.Text(r"\W+"))


def explain_with_lime(
    text: str,
    predict_fn,
    class_names: list,
    num_features: int = 6,
    num_samples: int = 5000,
    explainer: LimeTextExplainer = None,
) -> dict:
    if explainer is None:
        explainer = create_lime_explainer(class_names)
    explanation = explainer.explain_instance(
        text, predict_fn, num_features=num_features, num_samples=num_samples
    )
    return dict(explanation.as_list())


def explain_with_shap(
    text: str,
    predict_fn,
    num_features: int = 6,
    explainer: shap.Explainer = None,
) -> dict:
    if explainer is None:
        explainer = create_shap_explainer(predict_fn)
    shap_values = explainer([text])
    tokens      = shap_values.data[0]
    pred_class  = int(predict_fn([text]).argmax(axis=1)[0])
    values      = shap_values.values[0][:, pred_class]
    top         = sorted(zip(tokens, values), key=lambda x: abs(x[1]), reverse=True)[:num_features]
    return {token.strip(): float(score) for token, score in top}


def explain_prediction(text: str, predict_fn, class_names: list) -> dict:
    return {
        "lime": explain_with_lime(text, predict_fn, class_names),
        "shap": explain_with_shap(text, predict_fn),
    }
