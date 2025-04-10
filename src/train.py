#Importing the necessary libraries
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

#Importing the dataset
df = pd.read_csv("../Datasets/train.csv")

#Features and Label Split
X = df['Content']
y = df['Label']

#Creating a pipeline for the model

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2))),
    ('clf', MultinomialNB(alpha = 0.1))
])

#Fitting the pipeline to the training data
pipeline.fit(X, y)
print("Pipeline fitted to training data.")

# Save full pipeline
joblib.dump(pipeline, '../models/hate_speech_pipeline.joblib')
print('Pipeline saved as hate_speech_pipeline.joblib')
