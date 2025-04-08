#Importing the necessary libraries
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
from sklearn.pipeline import Pipeline
import joblib

#Importing the dataset
df = pd.read_csv("../Datasets/train.csv")

#Checking for the most common words in the dataset from the content column which are lablled as 1 (hatespeech)
hate_df = df[df['Label'] == 1]

#Combine all content into one string

hate_text = " ".join(hate_df['Content'].tolist())

text = re.sub(r'[^A-Za-z! ]+', '', hate_text)

# Generate WordCloud
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white',
    stopwords=ENGLISH_STOP_WORDS,
    colormap='Reds'
).generate(text)

#Selecting the top 30 most common hatewords in the dataset
hatewords = list(wordcloud.words_.keys())[:30]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(df['Content'], df['Label'], test_size=0.2, random_state=42, shuffle=True)

# Initialize TF-IDF normally first
tfidf = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Get feature names
feature_names = tfidf.get_feature_names_out()

# Find indices of hatewords in feature matrix
hateword_indices = [i for i, word in enumerate(feature_names) if word in hatewords]

# Boost TF-IDF weights of hatewords
boost_factor = 2.0 # You can tune this
X_train_tfidf_boosted = X_train_tfidf.copy()
X_train_tfidf_boosted[:, hateword_indices] *= boost_factor

X_test_tfidf_boosted = X_test_tfidf.copy()
X_test_tfidf_boosted[:, hateword_indices] *= boost_factor

# Train MultinomialNB
clf = MultinomialNB(alpha = 0.1)
clf.fit(X_train_tfidf_boosted, y_train)
y_pred = clf.predict(X_test_tfidf_boosted)

# Results
print(classification_report(y_test, y_pred))

#Save the MultinomialNB model to a file and the TF-IDF vectorizer to a file
joblib.dump(clf, '../models/hate_speech_model.pkl')
print('Model saved as hate_speech_model.pkl')

joblib.dump(tfidf, '../models/tfidf_vectorizer.pkl')
print('TF-IDF vectorizer saved as tfidf_vectorizer.pkl')
