# 📧 Spam Mail Classifier

A clean, production-ready spam email classifier using Naive Bayes algorithm with **98.85% accuracy**.

## 🎯 Features

- **Single, Optimized Model**: Naive Bayes (98.85% accuracy)
- **Simple & Fast**: Minimal dependencies, quick predictions
- **Modern UI**: Clean, responsive interface
- **Easy Deployment**: Ready for Azure, Heroku, or any cloud platform

## 🚀 Quick Start

### 1. Train the Model

```bash
python train_model.py
```

This will:
- Load the spam.csv dataset
- Train a Naive Bayes classifier
- Save the model and vectorizer
- Test with example emails

### 2. Run the Web App

```bash
python app.py
```

Open http://localhost:5000 in your browser.

## 📊 Model Performance

- **Algorithm**: Multinomial Naive Bayes
- **Accuracy**: 98.85%
- **Training Data**: 5,572 emails (747 spam, 4,825 ham)
- **Features**: 7,489 unique words (CountVectorizer)

## 📁 Project Structure

```
SpamMailClassifer/
├── spam.csv                        # Dataset
├── SpamMail.ipynb                  # Original training notebook
├── train_model.py                  # Training script
├── app.py                          # Flask web application
├── static/
│   └── index.html                  # Frontend UI
├── spam_classifier_model.joblib    # Trained model
├── vectorizer.joblib               # Text vectorizer
├── requirements.txt                # Dependencies
└── README.md                       # This file
```

## 🔌 API Usage

### Classify Email

**POST** `/api/predict`

```json
{
  "email": "Your email text here"
}
```

**Response:**

```json
{
  "prediction": "Spam",
  "is_spam": true,
  "confidence": 95.67,
  "probability": {
    "ham": 0.0433,
    "spam": 0.9567
  }
}
```

## ☁️ Deployment

### Azure Web App

1. Create Azure Web App (Python 3.10+)
2. Deploy via Git or GitHub Actions
3. Set startup command: `gunicorn --bind=0.0.0.0:8000 app:app`

### Heroku

```bash
heroku create your-app-name
git push heroku main
```

## 🛠️ Tech Stack

- **Backend**: Flask
- **ML**: scikit-learn (Naive Bayes)
- **Frontend**: HTML, TailwindCSS, Vanilla JS
- **Deployment**: Gunicorn

## 📝 License

Open source - Educational purposes
