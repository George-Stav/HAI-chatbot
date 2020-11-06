from joblib import dump, load
from Handlers import objectFormatter
import pandas as pd
import numpy as np
from numpy import dot, linalg
import random as rand

data = pd.read_csv('../data/questions/dataset.csv')
answers = data['answer']

handle = objectFormatter('answer')

def sim(q, d):
    x = dot(q, d)
    y = linalg.norm(q)*linalg.norm(d)
    if y == 0:
        return y
    return x/y

def answer(query, debug = False):
    f_query = handle.format([query])
    q_query = handle.q(f_query, handle.vocab)

    similarity_index = [sim(q_query.tolist(), handle.bow[d].tolist()) for d in handle.bow] # create list of similarity indeces between query and bag-of-words

    max_sim = max(similarity_index)

    if max_sim == 0:
        print(snarky_response(query))
        return

    indeces = [i for i, x in enumerate(similarity_index) if x == max_sim] # find all indeces of answers with maximum similarity
    print(answers[rand.choice(indeces)]) # choose a random one out of the most fitting answers

    if debug:
        print('[' + str(round(max_sim, 3) * 100) + '%]')
        print(f_query)

def snarky_response(query):
    bots = [
        'Alexa',
        'Siri',
        'GoogleAssistant',
        'Cortana'
    ]

    bot = rand.choice(bots)

    responses = [
        'How am I supposed to know ' + query + '. What am I? ' + bot + "?" \
         if bot != 'Cortana' else bot + '? Actually who am I kidding, noone uses ' + bot + '.',
         'I don\'t feel like looking this up right now. You do it.',
         'You seriously want to know ' + query + '? I am disappointed...',
         'You seriously don\'t know ' + query + '? What a loser...'
    ]
    
    return rand.choice(responses)