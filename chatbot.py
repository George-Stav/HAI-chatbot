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
from nltk.stem.snowball import SnowballStemmer


########################
### GLOBAL VARIABLES ###
########################

USERNAME = None
BOTNAME = None
DEBUG = True


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

        chosen_intent = intent([userInput])
        answer(userInput, master[chosen_intent]['vocabulary'], master[chosen_intent]['bag'], master[chosen_intent]['answers'])
        

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

def answer(query, vocabulary, bag, answers):
    # answers = qna_dataset['answer']
    f_query = format([query])
    q_query = process_query(f_query, vocabulary)

    similarity_index = [cos_sim(q_query.tolist(), doc.tolist()) for doc in bag] # create list of similarity indeces between query and bag-of-words

    max_sim = max(similarity_index)

    # if max_sim == 0:
    #     bot_response(query)
    #     return

    indeces = [i for i, x in enumerate(similarity_index) if x == max_sim] # find all indeces of answers with maximum similarity
    random = rand.choice(indeces)
    # print(answers[random]) # choose a random one out of the most fitting answers

    if DEBUG:
        max_list = sorted(similarity_index, reverse=True)[:3]
        for m in max_list:
            index = similarity_index.index(m)
            print('[' + str(round(m, 3) * 100) + '] ==> ' + answers[index])
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
    stemmer = SnowballStemmer("english")
    a = []
    for doc in data:
        if re.search('\'', doc):
            doc = contractions.fix(doc) # remove cases with apostrophe (e.g. "I'm", "it's" etc.)
        t = tokenizer.tokenize(doc) # tokenize, remove punctuation
        a.append([stemmer.stem(x) for x in t])
        # a.append([x.lemma_.lower() for x in nlp(" ".join(t))])# if x.is_stop == False]) # lemmatize, remove stopwords and flatten uppercase
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
                if DEBUG:
                    print("\033[1;31;40mValueError: \'" + term + "\'\033[0m not in vocabulary. Term will be ignored.")
                continue
            bow[key][index] += 1
    return bow

def bow_list(data, vocabulary):
    bow = []
    for i, doc in enumerate(data):
        bow.append(np.zeros(len(vocabulary)))
        for term in doc:
            try:
                index = vocabulary.index(term)
            except ValueError:
                if DEBUG:
                    print("\033[1;31;40mValueError: \'" + term + "\'\033[0m not in vocabulary. Term will be ignored.")
                continue
            bow[i][index] += 1
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




from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

def apply_tfidf(data, vocabulary, variation="ds"):
    # pipe = Pipeline([('raw_term_frequency', CountVectorizer(vocabulary=vocabulary)), ('tfidf', TfidfTransformer())]).fit(data)
    countVect = CountVectorizer(vocabulary=vocabulary)

    return countVect.fit_transform(data)


def load_datasets():

    ###### QNA Dataset ######

    qna_dataset = pd.read_csv('./data/qna/dataset.csv')
    qna_category = "question"
    qna_path = f'./objects/qna/{qna_category}'

    try:
        qna_parsed = load(f'{qna_path}.parsed.joblib')
    except:
        print(f"Parsing qna[{qna_category}] dataset...")
        qna_parsed = format(qna_dataset[qna_category])
        dump(qna_parsed, f'{qna_path}.parsed.joblib')

    try:
        qna_vocabulary = load(f'{qna_path}.vocab.joblib')
    except:
        print("Creating vocabulary...")
        qna_vocabulary = vocab(qna_parsed)
        dump(qna_vocabulary, f'{qna_path}.vocab.joblib')

    try:
        qna_bag = load(f'{qna_path}.bow.joblib')
    except:
        print("Creating bag of words...")
        qna_bag = bow_list(qna_parsed, qna_vocabulary)
        dump(qna_bag, f'{qna_path}.bow.joblib')


    ###### Small Talk Dataset ######

    personality = "witty" # choose from ['witty', 'caring', 'enthusiastic', 'friendly', 'professional']
    variation = "ds" # choose from ['ds', 'dl'] referring to what is considered a document (ds -> section, dl -> line/question)
    extra = "" # any extra specifications; i.e. sw = stop words are not removed, stem = dataset is stemmed instead of lemmatised, sw_stem = combination of previous two
    questions, answers = read_small_talk_dataset(personality, variation)
    smltk_path = f'./objects/small_talk/{personality}_{variation}_{extra}'

    try:
        smltk_parsed = load(f'{smltk_path}.parsed.joblib')
    except:
        print(f"Parsing {personality}_{variation}_{extra} small talk dataset...")
        smltk_parsed = format(questions)
        dump(smltk_parsed, f'{smltk_path}.parsed.joblib')

    try:
        smltk_vocabulary = load(f'{smltk_path}.vocab.joblib')
    except:
        print("Creating vocabulary...")
        smltk_vocabulary = vocab(smltk_parsed)
        dump(smltk_vocabulary, f'{smltk_path}.vocab.joblib')

    try:
        smltk_bag = load(f'{smltk_path}.bow.joblib')
    except:
        print("Creating bag of words...")
        smltk_bag = bow_list(smltk_parsed, smltk_vocabulary)
        dump(smltk_bag, f'{smltk_path}.bow.joblib')

    return {
        "qna":{
            "parsed": qna_parsed,
            "vocabulary": qna_vocabulary,
            "bag": qna_bag,
            "answers": qna_dataset['answer']
        },
        "smltk":{
            "parsed": smltk_parsed,
            "vocabulary": smltk_vocabulary,
            "bag": smltk_bag,
            "answers": answers
        },
    }









import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from nltk.corpus import stopwords


def read_small_talk_dataset(personality, variation="ds"):
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

    answers = [x for x in answers if x] # remove empty entries

    if variation == "dl":
        answers = [a for a, q in zip(answers, questions.values()) for _ in range(len(q))] # each answer appears as many times as the documents (lines) it corresponds to
        questions = [question for section in questions.values() for question in section]
        # questions = {x:questions[i] for i, x in enumerate(answers)} #dl -> document = line
    else: #ds
        questions = [" ".join(questions[i]) for i in range(len(answers))]
        # questions = {x:" ".join(questions[i]) for i, x in enumerate(answers)} #ds -> document = section

    return questions, answers









































smltk_dataset, answers = read_small_talk_dataset("witty", "dl")
qna_csv = pd.read_csv('./data/qna/dataset.csv')

qna_dataset = [item for x in qna_csv if x != 'questionID' for item in qna_csv[x]] # and x != 'question'

# print("Small talk: {}".format(len(small_talk_dataset)))
# print("QnA: {}".format(len(qna_dataset['question'])))

labels = ["smltk" for _ in range(len(smltk_dataset))]
labels += ["qna" for _ in range(len(qna_dataset))]

combined_dataset = smltk_dataset + qna_dataset

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
    intent = classifier.predict(data_tfidf)[0]
    print(intent)
    return intent


from spacy import displacy

name_intent_keywords = [
    "name",
    "call",
    "my"
]


master = load_datasets()
start()
# print(master['qna']['bag'])





















