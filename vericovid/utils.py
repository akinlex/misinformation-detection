import re
import string
import pickle
from datetime import datetime
import pandas as pd
import numpy as np
from lexicalrichness import LexicalRichness
from textblob import TextBlob
from bs4 import BeautifulSoup

from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

loaded_model = pickle.load(open('vericovid/models/misinformation_pred_model.h5', 'rb'))
loaded_vec = pickle.load(open('vericovid/models/vectorizer.pickle', 'rb'))

# remove url
def remove_url(text):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split())

def remove_mentions(text):
    return ' '.join(re.sub("([@#][A-Za-z0-9_]+)|(\w+:\/\/\S+)"," ", text).split())

def clean_text(text):
    # Make text lowercase, remove text in square brackets, remove punctuation, HTML tags and remove words containing numbers.
    cleaned_text = BeautifulSoup(text, "html.parser").text
    cleaned_text = re.sub('[^a-zA-Z]', ' ', text)
    cleaned_text = re.sub('\[.*?\]', '', text)
    cleaned_text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    cleaned_text = re.sub('\w*\d\w*', '', text)
    
    # cleaned_text = cleaned_text.lower()

    # Tokenize the text into word
    words = nltk.word_tokenize(cleaned_text)

    #Remove stop words and words with length less than equal to 3
    filtered_words = [word for word in words if not word in stop_words and len(word) > 3]

    #Lemmatize
    output_sentence = ''
    for word in filtered_words:
        output_sentence = output_sentence  + ' ' + str(lemmatizer.lemmatize(word))

    return output_sentence

def pipeline(text, tf):
    X = tf.transform(text)
    return X