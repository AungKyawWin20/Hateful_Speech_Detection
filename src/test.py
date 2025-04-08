# test.py

import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the trained model
vectorizer = joblib.load('../models/tfidf_vectorizer.pkl')
model = joblib.load('../models/hate_speech_model.pkl')

# Load test data
df = pd.read_csv('../datasets/test.csv')  # Make sure test.csv has 'Content' and 'Label' columns
X_test = df['Content']
y_test = df['Label']
X_test = vectorizer.transform(X_test)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))