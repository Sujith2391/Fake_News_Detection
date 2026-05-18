
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
        # Check if the model is actually loaded
        if not hasattr(trainer, 'pipeline'):
            return jsonify({'error': 'Model is not trained or could not be loaded. Please run src/train.py first.'}), 500

        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        
        # Preprocess
        cleaned_text = processor.clean_text(text)
        
        # Predict using the robust model logic
        prediction_data = trainer.predict_with_confidence(cleaned_text)
        
        return jsonify(prediction_data)

    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
