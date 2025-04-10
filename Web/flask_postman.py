from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
pipeline = joblib.load('../models/hate_speech_pipeline.joblib') 

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    texts = data['text']
    predictions = pipeline.predict(texts)
    labels = []
    for p in predictions:
        if p == 0:
            labels.append("Not Hate Speech")
        else:
            labels.append("Hate Speech")
    
    return jsonify({"prediction": labels})

if __name__ == '__main__':
    app.run(debug=True)

"""

To run the Flask app, use the command: python flask_postman.py
Make sure to install Flask and joblib if you haven't already
pip install Flask joblib
You can test the API using Postmman or any other API testing tool

"""