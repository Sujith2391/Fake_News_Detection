
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

class ModelTrainer:
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 3), use_idf=True, smooth_idf=True)),
            ('clf', LogisticRegression(penalty="l2", C=1, solver='liblinear'))
        ])
        self.model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'final_model.pkl')

    def train(self, X_train, y_train):
        """Trains the model pipeline."""
        print("Training model...")
        self.pipeline.fit(X_train, y_train)
        print("Training complete.")

    def evaluate(self, X_test, y_test):
        """Evaluates the model and prints metrics."""
        print("Evaluating model...")
        predictions = self.pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, predictions))
        return accuracy

    def save_model(self):
        """Saves the trained model to disk."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.pipeline, f)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        """Loads the model from disk."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        with open(self.model_path, 'rb') as f:
            self.pipeline = pickle.load(f)
        print("Model loaded successfully.")

    def predict(self, text):
        """Predicts the label for a single text input (assumes text is already cleaned)."""
        return self.pipeline.predict([text])[0]
    
    def predict_proba(self, text):
        """Predicts probabilities."""
        return self.pipeline.predict_proba([text])[0]

    def predict_with_confidence(self, text):
        """Predicts the class and confidence score, mapping to REAL or FAKE safely."""
        prediction = self.predict(text)
        probabilities = self.predict_proba(text)
        confidence = float(max(probabilities))
        
        # Robustly interpret the prediction label
        pred_str = str(prediction).strip().upper()
        # In train.csv, true statements have label 'TRUE' (or boolean True), which means REAL news
        # FALSE or False means FAKE news
        if pred_str in ('1', 'TRUE', 'T', 'REAL'):
            result = "REAL"
            is_fake = False
        else:
            result = "FAKE"
            is_fake = True
            
        return {
            'result': result,
            'confidence': f"{confidence * 100:.2f}%",
            'is_fake': is_fake
        }
