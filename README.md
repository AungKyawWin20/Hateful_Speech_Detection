# Hate Speech Detection 

A machine learning-powered web application that detects hate speech in text using TF-IDF vectorization and Multinomial Naive Bayes classification. Available as both a web interface and POSTMAN API, with an additional Streamlit frontend for better user experience.

## 📊 Project Overview

This project uses a curated dataset from Kaggle to detect hate speech in text content. The model achieves 87% accuracy using a combination of TF-IDF vectorization and Multinomial Naive Bayes classification.

Dataset: [Hate Speech Detection Dataset (Kaggle)](https://www.kaggle.com/datasets/waalbannyantudre/hate-speech-detection-curated-dataset)

## 🚀 Features

- Multiple interfaces:
  - Web UI (Flask)
  - POSTMAN API (Flask)
  - Modern UI (Streamlit)
- Real-time text classification
- Simple and intuitive user experience
- Robust ML pipeline with TF-IDF and Naive Bayes
- API support for batch predictions

## 🧠 Technical Details

### Model Architecture
- **Vectorizer**: TF-IDF with n-grams (1,2)
- **Classifier**: Multinomial Naive Bayes (alpha=0.1)
- **Accuracy**: 87% on test set
- **Pipeline**: Scikit-learn Pipeline for seamless preprocessing and prediction

## 🛠️ Installation & Setup

1. Clone the repository:
```bash
git clone https://github.com/AungKyawWin20/Hateful_Speech_Detection.git
cd Hateful_Speech_Detection
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🏃‍♂️ Running the Applications

### 1. Flask Web Interface
```bash
python Web/HTML_App.py
```
Visit `http://localhost:5000` in your browser

### 2. POSTMAN
```bash
python Web/flask_postman.py
```

### 3. Streamlit Interface
```bash
cd Web/Streamlit
streamlit run app.py
```

## 📡 API Usage

### Endpoint: `/predict`
- Method: POST
- Content-Type: application/json

Example request:
```json
{
  "text": ["You fucking retard", "Thank you so much"]
}
```

Example response:
```json
{
  "prediction": [
    "Hate Speech",
    "Not Hate Speech"
  ]
}
```

## 📂 Project Structure
```
Hateful_Speech_Detection/
├── Datasets/
│   ├── HateSpeechDataset.csv
│   ├── HateSpeechDatasetBalanced.csv
│   ├── train.csv
│   └── test.csv
├── Web/
│   ├── HTML_App.py
│   ├── flask_postman.py
│   ├── Streamlit/
│   │   └── app.py
│   └── templates/
│       ├── index.html
│       └── result.html
├── models/
│   └── hate_speech_pipeline.joblib
├── src/
│   ├── train.py
│   └── test.py
├── requirements.txt
└── README.md
```

## 🔧 Development

To train a new model:
```bash
python src/train.py
```

To evaluate the model:
```bash
python src/test.py
```

## 🚀 Deployment

The application can be deployed on:
- Render
- Heroku
- AWS
- Google Cloud Platform

## 🛣️ Roadmap

- Implement BERT/Transformer-based models
- Improve model accuracy with larger dataset
- Add user feedback loop
- Implement model retraining pipeline

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 👏 Acknowledgments

- Dataset provided by Kaggle
- Built with Flask and Streamlit
- Powered by scikit-learn

## 📧 Contact

Project Link: [https://github.com/AungKyawWin20/Hateful_Speech_Detection](https://github.com/AungKyawWin20/Hateful_Speech_Detection)
