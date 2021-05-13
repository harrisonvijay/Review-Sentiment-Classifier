# Review Sentiment Classifier

- Classifies the entered text as Positive or Negative
- Check it out: https://rsc.harrisonvijay.repl.co/
- The website is responsive (adapts well to different screen sizes)

### Tools and frameworks used

- HTML, CSS, JS
- Flask, Jinja2
- pandas, sklearn, nltk packages

### Dataset

- IMDB Dataset of 50k movie reviews
- 25k Positive and 25k Negative
- Available at https://link-shr.herokuapp.com/imdb-dataset

### Process

- Training
  - Encoded the labels as 0 (negative) and 1 (positive)
  - Cleaned and lemmatized the review text
  - Vectorized the text using TF-IDF vectorizer
  - Used 70% of the dataset to train a logistic regression model
  - Tested it on the remaining 30% to get an idea of the accuracy (gave around 90%)
  - Then trained the model on the entire dataset
  - Stored the trained model and vectorizer using pickle (.pkl files)
- Server
  - Used Flask for the server
  - Loaded the trained model and vectorizer from the .pkl files
  - Upon GET request to the / route, a webpage which gets the text input is rendered
  - Upon GET request to the /classify route, the appropriate class/error is returned

### To make it work

- Install all the modules in requirements.txt (pip install -r requirements.txt)
- Download the dataset
- Run train.py to get the pickle files of the trained model and vectorizer
- Then run server.py
