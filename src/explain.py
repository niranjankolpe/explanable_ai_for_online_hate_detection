from lime.lime_text import LimeTextExplainer

class_names = ["NOT", "OFF"]

explainer = LimeTextExplainer(class_names=class_names)


def explain_prediction(text, model, vocab, predict_fn):

    explanation = explainer.explain_instance(
        text,
        predict_fn,
        num_features=6
    )

    return dict(explanation.as_list())