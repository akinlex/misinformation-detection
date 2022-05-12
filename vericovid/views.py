import os
import re
import requests
from django.shortcuts import render
from vericovid.utils import *

def auth():
    return os.getenv('TOKEN')

def create_headers(bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers

def connect_to_endpoint(url, headers, next_token = None):
    # params['next_token'] = next_token   #params object received from create_url function
    response = requests.request("GET", url, headers = headers)
    print("Endpoint Response Code: " + str(response.status_code))
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()

def predict(text):
    text_cleaned = remove_url(text)
    text_cleaned = clean_text(text_cleaned)    
    # text_cleaned = remove_mentions(text_cleaned)
    df = pd.DataFrame()
    df.loc[0,'cleaned_text'] = text_cleaned
    clean_text_new = pipeline(df['cleaned_text'], loaded_vec)
    prediction = loaded_model.predict(clean_text_new)

    for i in range(len(prediction)):
        if prediction[i] == 0:
            news = 'Misinformative'
        else:
            news = 'Informative'
    return { #return the dictionary for endpoint
         "ACTUAL STORY": text[:120],
         "PREDICTED": news,
         "PROBABILITY": loaded_model.predict_proba(clean_text_new)
    }

def index(request):
    url, res = None, None
    if request.method == 'POST':
        data = request.POST.dict()
        match = re.match(r'^https?:\/\/twitter\.com\/(?:#!\/)?(\w+)\/status(?:es)?\/(\d+)(?:\/.*)?$', data['tweet_url'])
        if match:
            url = data['tweet_url']
            tweet_id = re.findall(r'([0-9]+$)', data['tweet_url'])
            # tweet_by = re.findall(r'\/(?:#!\/)?(\w+)\/', data['tweet_url'])
            search_url = "https://api.twitter.com/2/tweets/"+tweet_id[0] #Change to the endpoint you want to collect data from

            bearer_token = auth()
            headers = create_headers(bearer_token)
            json_response = connect_to_endpoint(search_url, headers)
            text = json_response['data']['text']
            res = predict(text)
  
    return render(request, 'vericovid/index.html', context={'search_url':url, 'results':res})
