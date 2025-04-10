# Hateful_Speech_Detection
Developing a Flask-based web application for detecting hateful speech using a dataset from Kaggle

Link: https://www.kaggle.com/datasets/waalbannyantudre/hate-speech-detection-curated-dataset?select=HateSpeechDatasetBalanced.csv


A lightweight and accurate hate speech classifier powered by a TF-IDF vectorizer and Multinomial Naive Bayes. The app provides a web interface and a REST API for real-time predictions.

## 🚀 Features

- Text input interface to detect hate speech
- REST API endpoint for external usage (e.g., Postman, curl)
- Hateword-boosted TF-IDF vectorizer for improved recall
- Simple, aesthetic frontend with HTML + CSS
- Easily extendable and lightweight for deployment

## 🧠 Model Info

- **Vectorizer**: TF-IDF with n-grams (1,2)
- **Model**: Multinomial Naive Bayes (`alpha=0.1`)

## 🛠️ Installation & Setup

```bash
git clone https://github.com/AungKyawWin20/Hateful_Speech_Detection.git
pip install -r requirements.txt
```

To train the model:
```bash
python train.py
```

To test the model:
```bash
python test.py
```

## 🌐 Run the Flask Apps

For Web Interface
```bash
python HTML_App.py
```
Go to `http://localhost:5000` in your browser

For Postman
```bash
python flask_postman.py
```
Use `http://localhost:5000/predict` with POST JSON:

### Input

```json
{
  "text": ["You fucking retard", "Thank you so much"]
}
```

### Output

```json
{
  "prediction" : [
      "Hate Speech",
      "Not Hate Speech"
  ]
}
```

## Future Improvements
- Use a deep learning model like LSTM/SVM for improved accuracy
- Implement a more aestheic website design
- Host the app on Streamlit

## 📄 License
Apache 2.0 

