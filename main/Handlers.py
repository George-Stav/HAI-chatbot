import pandas as pd
from joblib import dump, load
import numpy as np
import nltk
import spacy
import sys

class objectFormatter:

    def __init__(self, category):
        self.loader = self.__objectLoader(self, category)
        self.matrix = self.loader.matrix
        self.vocab = self.loader.vocab
        self.bow = self.loader.bow

    def format(self, data):    
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

    def vocab(self, data):
        '''
        Create a vocabulary out of the data provided.
        '''
        vocabulary = []
        for document in data:
            for term in document:
                if term not in vocabulary:
                    vocabulary.append(term)
        return vocabulary

    def bow(self, keys, data, vocabulary):
        bow = {}
        for (key, document) in zip(keys, data):
            bow[key] = np.zeros(len(vocabulary))
            for term in document:
                try:
                    index = vocabulary.index(term)
                except ValueError:
                    print("\033[1;31;40mValueError: \'" + term + "\'\033[0m not in vocabulary. Term will be skipped.")
                    continue
                bow[key][index] += 1
        return bow

    def q(self, data, vocabulary):
        a = np.zeros(len(vocabulary))
        for term in data:
            try:
                index = vocabulary.index(term)
            except ValueError:
                print("\033[1;31;40mValueError: \'" + term + "\'\033[0m not in vocabulary. Term will be skipped.")
                continue
            a[index] += 1
        return a

    class __objectLoader:
        __validCategories = [
            'question',
            'answer',
            'document'
        ]

        def __init__(self, objFormatter, category):
            '''
            Category: Question, Answer, Document => 3 useable columns from provided Q&A dataset

            category: String
            '''
            if isinstance(category, str) == False:
                print("Error: Wrong type in objectLoader")
                sys.exit(1)

            category = category.lower()

            if category not in self.__validCategories:
                print("Warning: Invalid category name in objectLoader; Question will be used by default")
                category = "Question"

            self.CATEGORY = category
            self.OBJ_PATH = f'../objects/{category}'

            self.objFormatter = objFormatter
            self.dataset = pd.read_csv('../data/dataset.csv')
            self.qID = self.dataset['questionID']

            self.matrix = self.__load_matrix()
            self.vocab = self.__load_vocab()
            self.bow = self.__load_bow()

        def __load_matrix(self):
            try:
                matrix = load(f'{self.OBJ_PATH}.matrix.joblib')
            except:
                matrix = self.objFormatter.format(self.dataset[f'{self.CATEGORY}'])
                dump(matrix, f'{self.OBJ_PATH}.matrix.joblib')

            return matrix

        def __load_vocab(self):
            try:
                vocab = load(f'{self.OBJ_PATH}.vocab.joblib')
            except:
                vocab = self.objFormatter.vocab(self.matrix)
                dump(vocab, f'{self.OBJ_PATH}.vocab.joblib')

            return vocab

        def __load_bow(self):
            try:
                bow = load(f'{self.OBJ_PATH}.bow.joblib')
            except:
                bow = self.objFormatter.bow(self.qID, self.matrix, self.vocab)
                dump(bow, f'{self.OBJ_PATH}.bow.joblib')

            return bow

