from joblib import dump, load
from Handlers import objectFormatter
import pandas as pd
import numpy as np
from numpy import dot, linalg

data = pd.read_csv('../data/dataset.csv')
answers = data['answer']

handle = objectFormatter('question')

def sim(q, d):
    x = dot(q, d)
    y = linalg.norm(q)*linalg.norm(d)
    if y == 0:
        return y
    return x/y

def answer(query):
    f_query = handle.format([query])
    q_query = handle.q(f_query, handle.vocab)

    similarity_index = [sim(q_query.tolist(), handle.bow[d].tolist()) for d in handle.bow]

    max_sim = max(similarity_index)
    
    if max_sim == 0:
        return "How am I supposed to know " + query + ". What am I? Alexa?"

    index = similarity_index.index(max(similarity_index))
    return answers[index]