import shap
import numpy as np
from lime.lime_text import LimeTextExplainer

class_names = ["NOT", "OFF"]
explainer   = LimeTextExplainer(class_names=class_names)


def explain_with_lime(text, predict_fn):
    explanation = explainer.explain_instance(
        text,
        predict_fn,
        num_features=6
    )
    return dict(explanation.as_list())


def explain_with_shap(texts, predict_fn):
    shap_explainer = shap.Explainer(
        predict_fn,
        masker=shap.maskers.Text(r"\W+")
    )
    shap_values = shap_explainer(texts)
    result = {}
    tokens = shap_values.data[0]
    values = shap_values.values[0][:, 1]
    token_value_pairs = sorted(
        zip(tokens, values),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:6]
    for token, score in token_value_pairs:
        result[token] = float(score)
    return result


def explain_prediction(text, predict_fn):
    lime_explanation = explain_with_lime(text, predict_fn)
    shap_explanation = explain_with_shap([text], predict_fn)
    return {
        "lime": lime_explanation,
        "shap": shap_explanation
    }