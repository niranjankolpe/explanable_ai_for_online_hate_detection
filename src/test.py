# WARNING: Relative import errors may occur. Handle accordingly.
import predict
from predict import load_model

model, vocab = load_model()

label, conf = predict.predict("nice work bro", model, vocab)
print(label, conf)