
from flask import Flask, render_template, request, jsonify
from src.model_trainer import ModelTrainer
from src.data_processor import DataProcessor
import os

app = Flask(__name__)

# Load model and processor on startup
print("Initializing application...")
trainer = ModelTrainer()
processor = DataProcessor()

try:
    trainer.load_model()
except FileNotFoundError:
    print("Model not found. Please run training script first.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        
        # Preprocess
        cleaned_text = processor.clean_text(text)
        
        # Predict
        prediction_label = trainer.predict(cleaned_text)
        probabilities = trainer.predict_proba(cleaned_text)
        
        # Map label to class name (assuming 0=True, 1=False or vice versa based on training labels)
        # In LIAR dataset: 
        # Original mapping in DataPrep.py was erratic, but let's assume binary classification.
        # We need to know what the model output means.
        # Based on classifier.py: "Prediction of the News : Looking Fake News" if prediction[0] == 0 
        # This implies 0 is Fake, 1 is Real? Or vice-versa?
        # Let's check the training data labels. 
        # Wait, in the original 'front.py': if prediction[0] == 0: print("Fake") else: print("Real")
        # So 0 -> Fake, 1 -> Real.
        
        result = "REAL" if prediction_label == 1 else "FAKE"
        confidence = float(max(probabilities))
        
        return jsonify({
            'result': result,
            'confidence': f"{confidence * 100:.2f}%",
            'is_fake': int(prediction_label) == 0
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
