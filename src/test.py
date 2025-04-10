#Importing libraries
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Load the trained model
pipeline = joblib.load('../models/hate_speech_pipeline.joblib')

# Load test data
df = pd.read_csv('../datasets/test.csv') 
X_test = df['Content']
y_test = df['Label']

#Make Prediction
y_pred = pipeline.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

