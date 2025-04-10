from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
pipeline = joblib.load('../models/hate_speech_pipeline.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():
    text = request.form['user_text']
    prediction = pipeline.predict([text])[0]
    if prediction == 0:
        result = "Not Hate Speech"
    else:
        result = "Hate Speech"
    
    return render_template('result.html', prediction=result, user_text=text)

if __name__ == '__main__':
    app.run(debug=True)