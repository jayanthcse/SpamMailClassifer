"""
Train Spam Classifier - Single Model (Naive Bayes)
Based on the original SpamMail.ipynb notebook approach
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

print("="*60)
print("TRAINING SPAM CLASSIFIER (Naive Bayes)")
print("="*60)

# Load dataset
print("\n1. Loading spam.csv...")
df = pd.read_csv("spam.csv")
print(f"   ✓ Loaded {len(df)} emails")
print(f"\nFirst few rows:")
print(df.head())

# Convert Category to binary (spam=1, ham=0)
print("\n2. Converting labels...")
df['spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)
print(f"   ✓ Spam: {df['spam'].sum()}, Ham: {len(df) - df['spam'].sum()}")

# Train/test split
print("\n3. Creating train/test split (75/25)...")
x_train, x_test, y_train, y_test = train_test_split(
    df.Message, 
    df.spam, 
    test_size=0.25,
    random_state=42
)
print(f"   ✓ Training: {len(x_train)} samples")
print(f"   ✓ Testing: {len(x_test)} samples")

# Vectorize text using CountVectorizer
print("\n4. Vectorizing text with CountVectorizer...")
v = CountVectorizer()
x_train_count = v.fit_transform(x_train.values)
print(f"   ✓ Vocabulary size: {len(v.vocabulary_)} unique words")

# Train Naive Bayes model
print("\n5. Training Naive Bayes model...")
model = MultinomialNB()
model.fit(x_train_count, y_train)
print("   ✓ Model trained successfully")

# Test accuracy
print("\n6. Evaluating model...")
x_test_count = v.transform(x_test)
predictions = model.predict(x_test_count)
accuracy = accuracy_score(y_test, predictions)
print(f"   ✓ Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Test with example emails
print("\n7. Testing with example emails...")
test_emails = [
    'Hello there Ramu, Can we get together to watch F1 at my place tonight??',
    'Upto 20% discount on parking, exclusive offers just for you. Dont miss this reward!',
    'URGENT! You have won $1000. Click here to claim now!',
    'Hey, are we still meeting for lunch tomorrow?'
]

emails_count = v.transform(test_emails)
predictions = model.predict(emails_count)

for email, pred in zip(test_emails, predictions):
    label = "SPAM" if pred == 1 else "HAM"
    print(f"   [{label}] {email[:60]}...")

# Save model and vectorizer
print("\n8. Saving model and vectorizer...")
joblib.dump(model, 'spam_classifier_model.joblib')
joblib.dump(v, 'vectorizer.joblib')
print("   ✓ Saved: spam_classifier_model.joblib")
print("   ✓ Saved: vectorizer.joblib")

print("\n" + "="*60)
print("✓ TRAINING COMPLETE!")
print("="*60)
print("\nModel is ready to use!")
print("Next step: Create the web application")
