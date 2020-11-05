import pandas as pd
import numpy as np
import nltk
import spacy

nlp = spacy.load('en_core_web_sm') # spaCy english model
tokenizer = nltk.RegexpTokenizer(r'\w+') # use tokenizer that removes punctuation

def format(data):    
    '''
    Remove punctuation & stopwords, tokenize, flatten uppercase, lemmatize given documents

    data: Array of documents

    return: Array of arrays, each one containing formatted version of initial documents
    '''
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
            index = vocabulary.index(term)
            bow[key][index] += 1
    return bow

def q(data, vocabulary):
    a = np.zeros(len(vocabulary))
    for term in data:
        index = vocabulary.index(term)
        a[index] += 1
    return a