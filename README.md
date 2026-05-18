
# TruthLens: Advanced Fake News Detection System

**Live Demo**: [https://fake-news-detection-iota-one.vercel.app/](https://fake-news-detection-iota-one.vercel.app/)

TruthLens is a state-of-the-art web application designed to identify misinformation in news articles and headlines. Powered by Machine Learning and Natural Language Processing (NLP), it provides instant credibility scores with a premium, modern user interface.

## 🚀 Features

- **Real-Time Analysis**: Instantly predict if a news piece is "Real" or "Fake".
- **Confidence Scoring**: View the probability score of the prediction.
- **Premium UI**: Dark-themed, glassmorphic design with smooth animations.
- **Robust Backend**: Built with Flask and Scikit-learn, featuring a clean and modular architecture.

## 🛠️ Technology Stack

- **Frontend**: HTML5, CSS3 (Custom Glassmorphism), JavaScript (Fetch API)
- **Backend**: Python, Flask
- **Machine Learning**: Scikit-Learn, NLTK, Pandas, Pickle
- **Algorithm**: Logistic Regression with TF-IDF Vectorization

## 📂 Project Structure

```
/
├── app.py                  # Main Flask Entry Point
├── src/                    # Core Logic
│   ├── data_processor.py   # Text Cleaning & Preprocessing
│   └── model_trainer.py    # Training & Prediction Logic
├── templates/              # HTML Templates
│   └── index.html
├── static/                 # Static Assets (CSS/JS)
│   ├── css/
│   └── js/
└── models/                 # Serialized Models
    └── final_model.pkl
```

## ⚡ Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed.

### Installation

1. Clone the repository (or extract the files).
2. Install dependencies:
   ```bash
   pip install flask scikit-learn nltk pandas
   ```

### Running the App

1. **Train the Model** (First time only):
   ```bash
   python src/train.py
   ```
   This will process the data and create `models/final_model.pkl`.

2. **Start the Server**:
   ```bash
   python app.py
   ```

3. **Access the App**:
   Open your browser and navigate to `http://localhost:5000`.

## 📝 Usage

1. Copy a news headline or article text.
2. Paste it into the text area on the web page.
3. Click the **Analyze Content** button.
4. Review the verdict and confidence score.

## 🛡️ License

Private Project. All Rights Reserved.
