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
from sklearn.metrics import jaccard_score
import re, contractions
from itertools import zip_longest


########################
### GLOBAL VARIABLES ###
########################

QNA_CATEGORY = 'question'
DATA_DIR = './data'
OBJ_PATH = './objects'
QNA_OBJ_PATH = f'{OBJ_PATH}/questions/{QNA_CATEGORY}'
SMALL_TALK_OBJ_PATH = f'{OBJ_PATH}/small_talk'
USERNAME = None
BOTNAME = None
DEBUG = True

qna_dataset = pd.read_csv(f'{DATA_DIR}/qna/dataset.csv')

#################
### MAIN LOOP ###
#################

def start():
    prompt = lambda x: input(f'{x}> ')

    # USERNAME = prompt('What should I call you?\n')
    # BOTNAME = prompt('What is my name?')
    
    userInput = ''
    print('\nEnter \'q\' to quit any time.')

    while True:
        userInput = prompt('\nListening...')

        if userInput == '':
            continue
        elif userInput == 'q':
            break

        answer(userInput, v, b)
        # intent([userInput])

    print("Bye I guess.")


#################
### Questions ###
#################

def cos_sim(q, d):
    x = dot(q, d)
    y = linalg.norm(q)*linalg.norm(d)
    if y == 0:
        return y
    return x/y

def jaccard_sim(q, d):
    return jaccard_score(q, d, average='macro')

def answer(query, vocabulary, bag):
    # answers = qna_dataset['answer']
    _, answers = make_small_talk_dict("witty")

    f_query = format([query])
    q_query = process_query(f_query, vocabulary)

    similarity_index = [cos_sim(q_query.tolist(), bag[d].tolist()) for d in bag] # create list of similarity indeces between query and bag-of-words

    max_sim = max(similarity_index)

    # if max_sim == 0:
    #     bot_response(query)
    #     return

    indeces = [i for i, x in enumerate(similarity_index) if x == max_sim] # find all indeces of answers with maximum similarity
    random = rand.choice(indeces)
    print(answers[random]) # choose a random one out of the most fitting answers

    if DEBUG:
        print('[' + str(round(max_sim, 3) * 100) + '%]')
        print(f_query)
        # print(format([qna_dataset[QNA_CATEGORY][random]]))

def bot_response(query):
    ### snarky response ###
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
    
    # print(rand.choice(responses))

    ### normal response ###

    print("I can't answer your query right now.")


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
        if re.search('\'', doc):
            doc = contractions.fix(doc) # remove cases with apostrophe (e.g. "I'm", "it's" etc.)
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
    for (key, document) in zip_longest(keys, data, fillvalue=keys[0]):
        bow[key] = np.zeros(len(vocabulary))
        for term in document:
            try:
                index = vocabulary.index(term)
            except ValueError:
                if DEBUG:
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
            if DEBUG:
                print("\033[1;31;40mValueError: \'" + term + "\'\033[0m not in vocabulary. Term will be ignored.")
            continue
        a[index] += 1
    return a

def load_qna_parsed():
    try:
        p = load(f'{QNA_OBJ_PATH}.parsed.joblib')
    except:
        print(f"Parsing qna[{QNA_CATEGORY}] dataset...")
        p = format(qna_dataset[f'{QNA_CATEGORY}'])
        dump(p, f'{QNA_OBJ_PATH}.parsed.joblib')
    return p


def load_qna_vocab():
    try:
        v = load(f'{QNA_OBJ_PATH}.vocab.joblib')
    except:
        print("Creating vocabulary...")
        v = vocab(parsed_data)
        dump(v, f'{QNA_OBJ_PATH}.vocab.joblib')
    return v


def load_qna_bow():
    qID = qna_dataset['questionID']
    try:
        b = load(f'{QNA_OBJ_PATH}.bow.joblib')
    except:
        print("Creating bag of words...")
        b = bow(qID, parsed_data, vocabulary)
        dump(b, f'{QNA_OBJ_PATH}.bow.joblib')
    return b

parsed_data = load_qna_parsed()
vocabulary = load_qna_vocab()
bag = load_qna_bow()

# start()







def csv(data):
    df = pd.DataFrame(data)
    df.to_csv('./test.csv', index=False, encoding='UTF-8')






import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from nltk.corpus import stopwords


def make_small_talk_dict(personality):
    filename = f'./data/small_talk/chitchat_{personality}.qna'

    try:
        f = open(filename, 'r')
    except FileNotFoundError:
        print(f'{filename} not found.')
        return   
    
    questions = {x:[] for x in range(100)}
    answers = []

    index = 0

    for x in f:
        if x[0] == '-':
            questions[math.floor(index/2)].append(x[2:-1])
        elif x[0] == '`':
            line = f.readline()
            if line != '':
                answers.append(line[4:-1])
                index += 1
    
    f.close()

    answers = [x for x in answers if x]
    questions = {x:questions[i] for i, x in enumerate(answers)}
    # questions = {x:" ".join(questions[i]) for i, x in enumerate(answers)}

    return questions, answers

def load_small_talk_dataset(personality):
    questions_dict, answers = make_small_talk_dict(personality)

    try:
        p = load(f'{SMALL_TALK_OBJ_PATH}/{personality}.parsed.joblib')
    except:
        print(f"Parsing {personality} small talk dataset...")
        p = format(questions_dict.values())
        dump(p, f'{SMALL_TALK_OBJ_PATH}/{personality}.parsed.joblib')

    try:
        v = load(f'{SMALL_TALK_OBJ_PATH}/{personality}.vocab.joblib')
    except:
        print("Creating vocabulary...")
        v = vocab(p)
        dump(v, f'{SMALL_TALK_OBJ_PATH}/{personality}.vocab.joblib')

    try:
        b = load(f'{SMALL_TALK_OBJ_PATH}/{personality}.bow.joblib')
    except:
        print("Creating bag of words...")
        b = bow(answers, p, v)
        dump(b, f'{SMALL_TALK_OBJ_PATH}/{personality}.bow.joblib')

    return p, v, b

def load_small_talk_dataset_dl(personality):
    """Same as above but each line is considered a document instead of a group of questions (lines).

    Args:
         personality ([type]): [description]
    """
    questions_dict, answers = make_small_talk_dict(personality)

    try:
        p = load(f'{SMALL_TALK_OBJ_PATH}/{personality}_dl.parsed.joblib')
    except:
        print(f"Parsing {personality} small talk dataset...")
        p = [format(x) for x in questions_dict.values()]
        dump(p, f'{SMALL_TALK_OBJ_PATH}/{personality}_dl.parsed.joblib')

    try:
        v = load(f'{SMALL_TALK_OBJ_PATH}/{personality}_dl.vocab.joblib')
    except:
        print("Creating vocabulary...")
        p2 = [x for y in p for x in y]
        v = vocab(p2)
        dump(v, f'{SMALL_TALK_OBJ_PATH}/{personality}_dl.vocab.joblib')

    try:
        b = load(f'{SMALL_TALK_OBJ_PATH}/{personality}_dl.bow.joblib')
    except:
        print("Creating bag of words...")
        b = []
        for i, x in enumerate(p):
            b.append(bow([answers[i]], x, v))
        # dump(b, f'{SMALL_TALK_OBJ_PATH}/{personality}_dl.bow.joblib')

    return p, v, b

p, v, b = load_small_talk_dataset_dl("witty")

print(len(b[0]["Nah, I'm good."]))
print(len(v))
print(len(b))
print(p[0])

# test, _ = make_small_talk_dict("witty")



# start()







questions, answers = make_small_talk_dict("witty")


small_talk_dataset = [item for sublist in questions.values() for item in sublist] #[:math.floor(len(sublist)/2)]
questions_dataset = [item for x in qna_dataset if x != 'questionID' for item in qna_dataset[x]] # and x != 'question'

# print("Small talk: {}".format(len(small_talk_dataset)))
# print("QnA: {}".format(len(qna_dataset['question'])))

labels = ["small_talk" for _ in range(len(small_talk_dataset))]
labels += ["qna" for _ in range(len(questions_dataset))]

combined_dataset = small_talk_dataset + questions_dataset

# dict = {
#     'labels': labels,
#     'data': combined_dataset
# }

# df = pd.DataFrame(dict)
# df.to_csv('./test.csv', index=False, encoding='UTF-8')

# remember:
# x = data
# y = labels

SEED = 78054 #rand.randint(0, 100000)
x_train, x_test, y_train, y_test = train_test_split(combined_dataset, labels, stratify=labels, test_size=0.25, random_state=SEED)


countVect = CountVectorizer(stop_words=stopwords.words('english')) 
tfidf_transformer = TfidfTransformer(use_idf=True, sublinear_tf=True)

x_train_counts = countVect.fit_transform(x_train)
x_train_tf = tfidf_transformer.fit_transform(x_train_counts)

classifier = LogisticRegression(random_state=SEED).fit(x_train_tf, y_train)

x_test_counts = countVect.transform(x_test)
x_test_tf = tfidf_transformer.transform(x_test_counts)

predicted = classifier.predict(x_test_tf)

# print(confusion_matrix(y_test, predicted))
# print(accuracy_score(y_test, predicted))
# print(f1_score(y_test, predicted, pos_label='small_talk'))
# print("SEED: {}".format(SEED))


def intent(data):
    data_counts = countVect.transform(data)
    data_tfidf = tfidf_transformer.transform(data_counts)
    print(classifier.predict(data_tfidf))


from spacy import displacy

name_intent_keywords = [
    "name",
    "call",
    "my"
]

def test_format(data):
    tokenizer = nltk.RegexpTokenizer(r'\w+') # use tokenizer that removes punctuation
    nlp = spacy.load('en_core_web_sm') # spaCy english model    

    x = " be".join(re.split('\'s', data)) # replace all apostrophes with 'be', the base verb of "is", "are" etc.
    t = tokenizer.tokenize(x) # tokenize, remove punctuation
    doc = nlp(" ".join(t))
    displacy.render(doc)



# test = "What is my name?"

# test_format(test)






























# df1 = pd.DataFrame(load_qna_parsed())
# df2 = pd.DataFrame(load_qna_vocab())
# df3 = pd.DataFrame(load_qna_bow())

# df1.to_csv('./data/parsed.csv', index=False, encoding='UTF-8')
# df2.to_csv('./data/vocab.csv', index=False, encoding='UTF-8')
# df3.to_csv('./data/bow.csv', index=False, encoding='UTF-8')

# x = [max(bag[d]) for d in bag]

# for i in range(len(x)):
#     if i in x:
#         print("{}: {}".format(i, x.count(i)))
