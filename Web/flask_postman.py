from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('../models/hate_speech_model.pkl')
vectorizer = joblib.load('../models/tfidf_vectorizer.pkl')
@app.route('/predict', methods=['POST'])

def predict():
    data = request.get_json()
    content = data.get('text', '')
    
    # Vectorize the input text (transform it into numerical features)
    content_vectorized = vectorizer.transform([content])  # This creates a 2D array

    # Make the prediction
    prediction = model.predict(content_vectorized)[0]
    
    return jsonify({"prediction": int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)

"""

To run the Flask app, use the command: python flask_postman.py
Make sure to install Flask and joblib if you haven't already
pip install Flask joblib
You can test the API using Postmman or any other API testing tool

"""