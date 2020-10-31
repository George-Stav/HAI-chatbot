import pandas as pd
import nltk
import spacy

nlp = spacy.load('en_core_web_sm') # spaCy english model
tokenizer = nltk.RegexpTokenizer(r'\w+') # use tokenizer that removes punctuation

def format(data):    
    '''
    Remove punctuation & stopwords, tokenize, flatten uppercase, lemmatize given document

    data: Array of documents

    return: Array of arrays, each one containing formatted version of initial document
    '''
    a = []
    for d in data:
        t = tokenizer.tokenize(d) # tokenize, remove punctuation
        a.append([x.lemma_.lower() for x in nlp(" ".join(t)) if x.is_stop == False]) # lemmatize, remove stopwords and flatten uppercase
    return a