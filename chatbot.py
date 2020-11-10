###############
### IMPORTS ###
###############
import os, sys
from numpy import dot, linalg
import numpy as np
import random as rand
import pandas as pd
from joblib import dump, load
import spacy, nltk
import wikipedia

########################
### GLOBAL VARIABLES ###
########################

QNA_CATEGORY = 'question'
DATA_DIR = './data'
OBJ_PATH = './objects'
QNA_OBJ_PATH = f'{OBJ_PATH}/questions/{QNA_CATEGORY}'
USERNAME = None
BOTNAME = None

qna_dataset = pd.read_csv(f'{DATA_DIR}/qna/dataset.csv')

#################
### MAIN LOOP ###
#################

def start(debugMode = False):
    prompt = lambda x: input(f'{x}> ')

    USERNAME = prompt('What should I call you?\n')
    # BOTNAME = prompt('What is my name?')
    
    userInput = ''
    print('\nEnter \'q\' to quit any time.')

    while True:
        userInput = prompt('\nListening...')

        if userInput == '':
            continue
        elif userInput == 'q':
            break

        answer(userInput, debugMode)

    print("Bye I guess.")


#################
### Questions ###
#################

def sim(q, d):
    x = dot(q, d)
    y = linalg.norm(q)*linalg.norm(d)
    if y == 0:
        return y
    return x/y

def answer(query, debug = False):
    answers = qna_dataset['answer']

    f_query = format([query])
    q_query = process_query(f_query, vocabulary)

    similarity_index = [sim(q_query.tolist(), bag[d].tolist()) for d in bag] # create list of similarity indeces between query and bag-of-words

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


###############
### Helpers ###
###############


def format(data):    
    '''
    Remove punctuation & stopwords, tokenize, flatten uppercase, lemmatize given documents
    data: Array of documents
    return: Array of arrays, each one containing formatted version of initial documents
    '''
    tokenizer = nltk.RegexpTokenizer(r'\w+') # use tokenizer that removes punctuation
    nlp = spacy.load('en_core_web_sm') # spaCy english model    
    a = []
    for doc in data:
        t = tokenizer.tokenize(doc) # tokenize, remove punctuation
        a.append([x.lemma_.lower() for x in nlp(" ".join(t)) if x.is_stop == False]) # lemmatize, remove stopwords and flatten uppercase
    return a[0] if len(data) == 1 else a

    
def vocab(data):
    '''
    Create a vocabulary out of the data provided.
    '''
    vocabulary = []
    for document in data:
        for term in document:
            if term not in vocabulary:
                vocabulary.append(term)
    return vocabulary


def bow(keys, data, vocabulary):
    bow = {}
    for (key, document) in zip(keys, data):
        bow[key] = np.zeros(len(vocabulary))
        for term in document:
            try:
                index = vocabulary.index(term)
            except ValueError:
                print("\033[1;31;40mValueError: \'" + term + "\'\033[0m not in vocabulary. Term will be ignored.")
                continue
            bow[key][index] += 1
    return bow


def process_query(data, vocabulary):
    a = np.zeros(len(vocabulary))
    for term in data:
        try:
            index = vocabulary.index(term)
        except ValueError:
            print("\033[1;31;40mValueError: \'" + term + "\'\033[0m not in vocabulary. Term will be ignored.")
            continue
        a[index] += 1
    return a

def load_parsed():
    try:
        p = load(f'{QNA_OBJ_PATH}.parsed.joblib')
    except:
        print("Parsing qna dataset...")
        p = format(qna_dataset[f'{QNA_CATEGORY}'])
        dump(p, f'{QNA_OBJ_PATH}.parsed.joblib')
    return p


def load_vocab():
    try:
        v = load(f'{QNA_OBJ_PATH}.vocab.joblib')
    except:
        print("Creating vocabulary...")
        v = vocab(parsed_data)
        dump(v, f'{QNA_OBJ_PATH}.vocab.joblib')
    return v


def load_bow():
    qID = qna_dataset['questionID']
    try:
        b = load(f'{QNA_OBJ_PATH}.bow.joblib')
    except:
        print("Creating bag of words...")
        b = bow(qID, parsed_data, vocabulary)
        dump(b, f'{QNA_OBJ_PATH}.bow.joblib')
    return b

parsed_data = load_parsed()
vocabulary = load_vocab()
bag = load_bow()

start(debugMode=True)