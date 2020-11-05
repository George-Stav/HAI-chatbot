from joblib import dump, load
from format_document import format, vocab, bow, q
import pandas as pd
import numpy as np
from numpy import dot, linalg

data = pd.read_csv('./dataset.csv')
qID = data['QuestionID']
answers = data['Answer']
DOCUMENT = 'Question'
M_PATH = f'./joblib/{DOCUMENT}'
def sim(q, d):
    x = dot(q, d)
    y = linalg.norm(q)*linalg.norm(d)
    if y == 0:
        return y
    return x/y
    
# sim = lambda q, d: dot(q, d) / (linalg.norm(q)*linalg.norm(d))

formatted_data = []
vocabulary = []
bag = {}

try:
    formatted_data = load(f'{M_PATH}.matrix.joblib')
except:
    formatted_data = format(data[DOCUMENT])
    dump(formatted_data, f'{M_PATH}.matrix.joblib')

try:
    vocabulary = load(f'{M_PATH}.vocab.joblib')
except:
    vocabulary = vocab(formatted_data)
    dump(vocabulary, f'{M_PATH}.vocab.joblib')

try:
    bag = load(f'{M_PATH}.bow.joblib')
except:
    bag = bow(qID, formatted_data, vocabulary)
    dump(bag, f'{M_PATH}.bow.joblib')

def answer(query):
    f_query = format([query])

    print(f_query)

    q_query = q(f_query, vocabulary)



    similarity_index = [sim(q_query.tolist(), bag[d].tolist()) for d in bag]

    max_sim = max(similarity_index)
    
    if max_sim == 0:
        return "How am I supposed to know " + query + ". What am I? Alexa?"

    index = similarity_index.index(max(similarity_index))
    return answers[index]

answer("I don't know what to ask you")