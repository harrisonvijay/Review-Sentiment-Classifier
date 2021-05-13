import re
import string
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import nltk

nltk.download("stopwords")
nltk.download("wordnet")

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

# Read Dataset
data = pd.read_csv("IMDB Dataset.csv")

# Encode the labels - negative and positive as 0 and 1
data["sentiment"] = LabelEncoder().fit_transform(data["sentiment"])

# Clean the text
def clean(text):
    text = text.lower()
    text = re.sub("\[.*?\]", " ", text)
    text = re.sub("<.*?>+", " ", text)
    text = re.sub("\\W", " ", text)
    text = re.sub("https?://\S+|www\.\S+", " ", text)
    text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub("\n", "", text)
    text = re.sub("\w*\d\w*", "", text)
    text = re.sub("[^a-zA-Z]", " ", text)
    return text


data["review"] = data["review"].apply(clean)

# Lemmatize each word in the text
stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()
corpus = []
for idx in range(len(data)):
    text = data["review"][idx]
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    text = " ".join(lemmatized_words)
    corpus.append(text)

# Vectorizing the text using
# Term frequency - Inverse Document Frequency (TF-IDF) vectorizer
tfidf = TfidfVectorizer(ngram_range=(1, 3))
X = tfidf.fit_transform(corpus)
y = data["sentiment"]

# Train a Logistic regression model with 70% of the dataset and
# test with the remaining 30% to get an idea of the accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
clf = LogisticRegression().fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

# Train with 100% of the dataset
clf = LogisticRegression().fit(X, y)

# Serialize the model & vectorizer and store them
pickle.dump(clf, open("sentiment_model.pkl", "wb"))
pickle.dump(tfidf, open("sentiment_vectorizer.pkl", "wb"))