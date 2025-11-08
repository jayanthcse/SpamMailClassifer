"""
Spam Mail Classifier Web Application
Simple Flask app with single Naive Bayes model
"""
from flask import Flask, request, jsonify, send_from_directory
import joblib
import os

app = Flask(__name__, static_folder='static')

# Add CORS headers for local testing
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

# Load model and vectorizer
print("Loading model and vectorizer...")
model = joblib.load('spam_classifier_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')
print("✓ Model loaded successfully!")

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        data = request.get_json()
        email_text = data.get('email', '')
        
        if not email_text:
            return jsonify({'error': 'No email text provided'}), 400
        
        # Vectorize and predict
        email_vectorized = vectorizer.transform([email_text])
        prediction = model.predict(email_vectorized)[0]
        
        # Get probability
        proba = model.predict_proba(email_vectorized)[0]
        probability = {
            'ham': float(proba[0]),
            'spam': float(proba[1])
        }
        
        result = {
            'prediction': 'Spam' if prediction == 1 else 'Ham',
            'is_spam': bool(prediction),
            'probability': probability,
            'confidence': float(max(proba)) * 100
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model': 'Naive Bayes'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n{'='*60}")
    print(f"🚀 Spam Mail Classifier Running!")
    print(f"📧 Open: http://localhost:{port}")
    print(f"{'='*60}\n")
    app.run(host='0.0.0.0', port=port, debug=True)
