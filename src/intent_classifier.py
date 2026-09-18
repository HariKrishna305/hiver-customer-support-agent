import joblib


class IntentClassifier:

    def __init__(self, model_path="models/intent_classifier.pkl"):

        print("Loading intent classifier...")

        self.model = joblib.load(model_path)

        print("Intent classifier loaded successfully.")

    def predict(self, customer_message):

        intent = self.model.predict(
            [customer_message]
        )[0]

        return intent