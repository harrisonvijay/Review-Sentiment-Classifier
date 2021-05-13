import os
import re
import string
import pickle
from flask import Flask, request, render_template
import nltk

# If not already downloaded
# nltk.download("stopwords")
# nltk.download("wordnet")

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

app = Flask(__name__, template_folder="templates")

lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words("english")

# Open the pickle files and deserialize
# to get the model and vectorizer objects
clf = pickle.load(open("sentiment_model.pkl", "rb"))
tfidf = pickle.load(open("sentiment_vectorizer.pkl", "rb"))


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


# Returns the sentiment of the text and confidence score
# Returns [-1, -1] if the confidence is less than 65%
# or if the lemmatized sentence is empty
def classifyText(text):
    text = clean(text)
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    text = " ".join(lemmatized_words)
    if len(text) == 0:
        return [-1, -1]
    ip = tfidf.transform([text])
    op_class = clf.predict(ip)[0]
    confidenceScore = clf.predict_proba(ip)[0][op_class]
    if confidenceScore * 100 < 65:
        return [-1, -1]
    return [op_class, confidenceScore]


# Render index.html upon GET request to /
@app.route("/")
def home():
    return render_template("index.html")


# Classify the text passed as URL parameter
# upon GET request to /classify
@app.route("/classify")
def classify():
    text = request.args.get("text")
    if text == None or len(text) == 0:
        return ({"error": "Text cannot be empty"}), 400
    res = classifyText(text)
    if res[0] == -1:
        return ({"error": "Not enough context to classify"}), 400
    elif res[0] == 1:
        return {"sentiment": "Positive", "confidence_score": res[1]}
    else:
        return {"sentiment": "Negative", "confidence_score": res[1]}


# Start the server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(port=port)