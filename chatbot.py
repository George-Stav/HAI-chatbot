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
PERSONALITY = 'witty'

#################
### MAIN LOOP ###
#################

def start():
    prompt = lambda x: input(f'{x}> ')

    USERNAME = prompt('What should I call you?\n').capitalize()
    # BOTNAME = prompt('What is my name?')

    userInput = ''
    print('\nEnter \'q\' to quit any time.')

    while True:
        userInput = prompt('\nListening...')

        if userInput == '':
            continue
        elif userInput == 'q':
            break
        elif userInput == 'run test queries':
            run_test_queries()
            continue

        name = name_intent(userInput)

        if name:
            if name == "recall":
                print("You told me your name is " + USERNAME + ". Anything else?")
                continue
            else:
                print("Noted " + name + ".")
                USERNAME = name
                continue

        intent = get_intent([userInput])
        answer(userInput, intent, verbose=True, tfidf=True)

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


def answer(query, intent, silent=False, verbose=False, tfidf=True):

    if not tfidf:
        query = process_query(format([query]), master[intent]['vocabulary'])
        similarity_index = [cos_sim(query.tolist(), doc.tolist()) for doc in master[intent]['bag']]
    else:
        query = apply_tfidf([query], master[intent]['vocabulary']).toarray()
        similarity_index = [cos_sim(query.tolist(), doc.tolist()) for doc in master[intent]['tfidf'].toarray()]
    
    top_results = sorted(similarity_index, reverse=True)[:3]
    top_results = {i:x for i, x in enumerate(similarity_index) if x in top_results and x != 0}

    if silent:
        return top_results
    
    if verbose:
        for i, x in zip(top_results.keys(), top_results.values()):
            try:
                x = x[0]
            except:
                NotImplemented
            print('[' + str(round(x, 3) * 100) + '] ==> ' + master[intent]['answers'][i])
    else: # choose a random answer out of the top results
        # indeces = [i for i, x in enumerate(similarity_index) if x == max(top_results)]
        print(master[intent]['answers'][rand.choice(top_results.keys())])


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


def format(data, keep_stop=False, stem=False, detect_name=False):
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

        if detect_name:
            f_doc = nlp(" ".join(t))
            names = [x.text for x in f_doc.ents]
            if names:
                if DEBUG: # print all names it found
                    print("DEBUG message")
                    print(names)
                return names # [0]
            return [x.lemma_.lower() for x in f_doc if not x.is_stop] # return original data without stop words if no entities found
            
        if stem:
            a.append([stemmer.stem(x) for x in t if x not in stopwords.words('english') or keep_stop])
        else:
            a.append([x.lemma_.lower() for x in nlp(" ".join(t)) if not x.is_stop or keep_stop])# lemmatize, remove stopwords and flatten uppercase
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

def apply_tfidf(unparsed_data, vocabulary):
    # pipe = Pipeline([('raw_term_frequency', CountVectorizer(vocabulary=vocabulary)), ('tfidf', TfidfTransformer())]).fit(data)
    countVect = CountVectorizer(vocabulary=vocabulary)#, analyzer=format)
    tfidf_transformer = TfidfTransformer(sublinear_tf=True)
    counts = countVect.fit_transform(unparsed_data)
    return tfidf_transformer.fit_transform(counts)



def load_datasets():

    ###### Formatting Options ######

    sw = False
    stem = False

    ###### QNA Dataset ######

    # special categories:
    ### all => 3 meaningful columns stacked on top of each other
    ### combined => concatenation of 2 or more meaningful categories
    ###             number of documents stays the same but length of each document increases
    ###             QD => question + document

    qna_dataset = pd.read_csv('./data/qna/dataset.csv')
    qna_category = "question"
    extra = "" # any extra specifications; i.e. sw = stop words are not removed, stem = dataset is stemmed instead of lemmatised, sw_stem = combination of previous two
    qna_path = f'./objects/qna/{qna_category}_{extra}'
    all_categories = qna_dataset['question'].tolist() + qna_dataset['answer'].tolist() + qna_dataset['document'].tolist()
    all_categories_answers = qna_dataset['answer'].tolist() + qna_dataset['answer'].tolist() + qna_dataset['answer'].tolist()
    combined_categories = [x+y for x,y in zip(qna_dataset['question'], qna_dataset['answer'])]

    if qna_category == 'all':
        qna_unparsed = all_categories
    elif re.search('combined', qna_category):
        qna_unparsed = combined_categories
    else:
        qna_unparsed = qna_dataset[qna_category]


    if re.search('sw', extra):
        sw = True
    if re.search('stem', extra):
        stem = True

    try:
        qna_parsed = load(f'{qna_path}.parsed.joblib')
    except:
        print(f"Parsing qna[{qna_category}] dataset...")
        qna_parsed = format(qna_unparsed, stem=stem, keep_stop=sw)
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

    try:
        qna_tfidf = load(f'{qna_path}.tfidf.joblib')
    except:
        print("Creating tfidf matrix...")
        qna_tfidf = apply_tfidf(qna_unparsed, qna_vocabulary)
        dump(qna_tfidf, f'{qna_path}.tfidf.joblib')


    ###### Small Talk Dataset ######

    personality = PERSONALITY # choose from ['witty', 'caring', 'enthusiastic', 'friendly', 'professional']
    variation = "ds" # choose from ['ds', 'dl'] referring to what is considered a document (ds -> section, dl -> line/question)
    extra = "" # same as above
    questions, answers = read_small_talk_dataset(personality, variation)
    smltk_path = f'./objects/small_talk/{personality}_{variation}_{extra}'

    if re.search('sw', extra):
        sw = True
    if re.search('stem', extra):
        stem = True

    try:
        smltk_parsed = load(f'{smltk_path}.parsed.joblib')
    except:
        print(f"Parsing {personality}_{variation}_{extra} small talk dataset...")
        smltk_parsed = format(questions, stem=stem, keep_stop=sw)
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

    try:
        smltk_tfidf = load(f'{smltk_path}.tfidf.joblib')
    except:
        print("Creating tfidf matrix...")
        smltk_tfidf = apply_tfidf(questions, smltk_vocabulary)
        dump(smltk_tfidf, f'{smltk_path}.tfidf.joblib')

    return {
        "qna":{
            "unparsed": qna_unparsed,
            "parsed": qna_parsed,
            "vocabulary": qna_vocabulary,
            "bag": qna_bag,
            "tfidf": qna_tfidf,
            "answers": qna_dataset['answer'] if qna_category != 'all' else all_categories_answers,
            "sw": sw,
            "stem": stem
        },
        "smltk":{
            "unparsed": questions,
            "parsed": smltk_parsed,
            "vocabulary": smltk_vocabulary,
            "bag": smltk_bag,
            "tfidf": smltk_tfidf,
            "answers": answers,
            "sw": sw,
            "stem": stem
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
        print(f'{filename} not found. Defaulting to witty personality.')
        filename = './data/small_talk/chitchat_witty.qna'
    
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
    elif variation == "ds":
        questions = [" ".join(questions[i]) for i in range(len(answers))]
        # questions = {x:" ".join(questions[i]) for i, x in enumerate(answers)} #ds -> document = section
    else: # special variation
        answers = [a for a, q in zip(answers, questions.values()) for _ in range(len(q))] # same as 'dl'
        questions = [section for section in questions.values()] # [[section_0], [section_1], ..., [section_n]]

    return questions, answers


def init_classifier(SEED=rand.randint(1,100000), print_evaluation=False):
    rand.seed(SEED)

    personalities = [
        'witty',
        'caring',
        'professional',
        'friendly',
        'enthusiastic'
    ]

    qna_csv = pd.read_csv('./data/qna/dataset.csv')
    qna_dataset = [item for x in qna_csv if x != 'questionID' for item in qna_csv[x]] # and x != 'question'
    # smltk_dataset = [item for section in read_small_talk_dataset('witty', 'special')[0] for i, item in enumerate(section) if i <= len(section)/2]
    # full_smltk_dataset, _ = read_small_talk_dataset(rand.choice(personalities), 'dl')
    smltk_dataset, _ = read_small_talk_dataset(PERSONALITY, 'dl')

    # full_smltk_dataset = [x for p in personalities for x in read_small_talk_dataset(p, 'dl')[0]]  
    # smltk_dataset = []
    # while len(smltk_dataset) != len(qna_dataset):
    #     x = rand.choice(full_smltk_dataset)
    #     if x not in smltk_dataset:
    #         smltk_dataset.append(x)

    print("Small talk: {}".format(len(smltk_dataset)))
    print("QnA: {}".format(len(qna_dataset)))

    labels = ["smltk" for _ in range(len(smltk_dataset))]
    labels += ["qna" for _ in range(len(qna_dataset))]

    combined_dataset = smltk_dataset + qna_dataset

    # remember:
    # x = data
    # y = labels
    x_train, x_test, y_train, y_test = train_test_split(combined_dataset, labels, stratify=labels, test_size=0.25, random_state=SEED)

    x_train_counts = countVect.fit_transform(x_train)
    x_train_tf = tfidf_transformer.fit_transform(x_train_counts)

    classifier = LogisticRegression(random_state=SEED).fit(x_train_tf, y_train)

    x_test_counts = countVect.transform(x_test)
    x_test_tf = tfidf_transformer.transform(x_test_counts)

    predicted = classifier.predict(x_test_tf)

    if print_evaluation:
        print(confusion_matrix(y_test, predicted))
        print(accuracy_score(y_test, predicted))
        print(f1_score(y_test, predicted, pos_label='smltk'))
        print("SEED: {}".format(SEED))

    return classifier

def get_intent(data, silent=False):
    data_counts = countVect.transform(data)
    data_tfidf = tfidf_transformer.transform(data_counts)
    intent = classifier.predict(data_tfidf)[0]
    if not silent:
        print(intent)
    return intent


def run_test_queries():
    f = open('./data/qna/test_queries.txt', 'r')

    for line in f:
        print("\n~~~~~ " + line[:-1] + " ~~~~~\n")
        query = line[line.find(':')+2:-1]
        intent = get_intent([query])
        answer(query, intent, verbose=True, tfidf=True)

    f.close()

def test_qna():
    """Test the qna 
    """
    qna_dataset = pd.read_csv('./data/qna/dataset.csv')
    questions = qna_dataset['question']
    answers = qna_dataset['answer']

    misclassified = {}
    correct = {}
    for i, q in zip(enumerate(questions), range(100)):
        intent = get_intent([q], silent=True)
        if intent == 'smltk':
            misclassified['Q' + str(i+1)] = q
            continue
        
        results = answer(q, intent, silent=True)
        correct[q] = ['Q'+str(i+1) for i in results if results[i] == 100]

    print(misclassified)
# questions, answers = read_small_talk_dataset("witty", variation="ds")

def name_intent(query):
    recall_vocabulary = [
        'name',
        'my',
        'say',
        'call',
        'what',
        'is'
    ]

    change_vocabulary = [
        'name',
        'to',
        'say',
        'call',
        'change',
        'me',
        'my'
    ]


    f_query = format([query], keep_stop=True, stem=True)

    binary_recall = [False for _ in range(len(recall_vocabulary))]
    binary_change = [False for _ in range(len(change_vocabulary))]

    for i, word in enumerate(format(recall_vocabulary, keep_stop=True, stem=True)):
        if word[0] in f_query and not binary_recall[i]:
            binary_recall[i] = True

    for i, word in enumerate(format(change_vocabulary, keep_stop=True, stem=True)):
        if word[0] in f_query and not binary_change[i]:
            binary_change[i] = True

    result = ""
    if binary_recall.count(True) > binary_change.count(True):
        if binary_recall.count(True) >= 2:
            return "recall"
    elif binary_change.count(True) > binary_recall.count(True):
        if binary_change.count(True) >= 2:
            result = format([query], detect_name=True)
            if not len(result)==1:
                result = [x for x in result if x not in change_vocabulary]
    
    # print('name_intent:')
    # if isinstance(result, list) and len(result)==1:
    #     print('here')
    #     print(result[0])
    # else:
    #     print("there")
    #     print(result)
    # print(result[0] if isinstance(result, list) and len(result)==1 else result)
    return result[0].capitalize() if isinstance(result, list) and len(result)==1 else " ".join([x.capitalize() for x in result])



countVect = CountVectorizer(stop_words=stopwords.words('english')) 
tfidf_transformer = TfidfTransformer(use_idf=True, sublinear_tf=True)
classifier = init_classifier(SEED=78054, print_evaluation=True)
master = load_datasets()



start()
# testString = "Call me george"
# print(format([testString]))
# print(name_intent(testString))




