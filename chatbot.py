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
import re, contractions
from itertools import zip_longest
from nltk.stem.snowball import SnowballStemmer
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

########################
### GLOBAL VARIABLES ###
########################

DEBUG = True # DEBUG mode prints more things relating to the performance of the system; e.g. thresholds are not taken into account when giving a response
USERNAME = None
PERSONALITY = None
QNA_THRESHOLD = 0.58
SMLTK_THRESHOLD = 0.76 # change to 0.24 if classifier was trained with ds instead of dl
RECALL_THRESHOLD = 0.34
CHANGE_THRESHOLD = 0.25

prompt = lambda x: input(f'{x}> ')

#################
### MAIN LOOP ###
#################

def start():
    """Main loop between bot and user. Starts by asking for the user's name and exits by pressing 'q'.
    """

    if DEBUG:
        print("\033[1;31;40m ~~~ DEBUG MODE IS ENABLED ~~~ \033[0m")

    global USERNAME
    USERNAME = prompt('What should I call you?\n').capitalize()

    userInput = ''
    print('\nEnter \'q\' to quit any time.')

    while True:
        userInput = prompt('\nListening...')

        # Special cases
        if userInput == '':
            continue
        # Quit
        elif userInput == 'q':
            break
        # Run provided test queries
        elif userInput == 'run test queries':
            run_test_queries()
            continue
        # Run altered test queries containing typos
        elif userInput == 'run test queries with typos':
            run_test_queries(typos=True)
            continue

        name = name_intent(userInput)
        if name:
            continue
        intent = qna_smltk_intent(userInput, silent=True)
        qna_smltk_answer(userInput, intent, verbose=False, tfidf=True)

    print("Talk to you later.")


#########################
### QNA/SMLTK Intents ###
#########################

def qna_smltk_answer(query, intent, verbose=False, tfidf=True):
    """Computes similarity index between provided user query and loaded datasets. Answers the user based on their query.

    Args:
        query (str): User query.
        intent (str): Detected intent:
            - "smltk"
            - "qna"
        verbose (bool, optional): Prints top 3 results. Defaults to False.
        tfidf (bool, optional): Use bag of words with tfidf weighting. Defaults to True.
    """
    f_query = format(query, sw=master[intent]['sw'], stem=master[intent]['stem'])

    if tfidf:
        query = apply_tfidf(f_query, master[intent]['vocabulary']).toarray()
        similarity_index = [cos_sim(query.tolist(), doc.tolist()) for doc in master[intent]['tfidf'].toarray()]
    else:
        query = bow(f_query, master[intent]['vocabulary']) 
        similarity_index = [cos_sim(query, doc.tolist()) for doc in master[intent]['bag']]
    
    top_results = sorted(similarity_index, reverse=True)[:3]
    top_results = {i:x for i, x in enumerate(similarity_index) if x in top_results and x != 0}
    top_results = dict(sorted(top_results.items(), key=lambda item: item[1], reverse=True))

    if DEBUG:
        if top_results:
            print("-formatted query => [" + ", ".join(f_query) + "]")
        else:
            default_answer(intent)
    else:
        if not top_results:
            default_answer(intent)
            return
        # Give default answers when most similar result doesn't exceed the required threshold
        if intent == 'smltk' and max(top_results.values()) < SMLTK_THRESHOLD:
            default_answer("smltk")
            return
        if intent == 'qna' and max(top_results.values()) < QNA_THRESHOLD:
            default_answer("qna")
            return

    if verbose:
        for key, item in top_results.items():
            try:
                item = item[0]
            except:
                NotImplemented
            print('[' + str(round(item, 3) * 100) + '] ==> ' + master[intent]['answers'][key] + \
                    " :: \"" + master[intent]['unparsed'][key] + "\" :: [" + ", ".join(master[intent]['parsed'][key]) + "]")
    else: # choose a random answer out of the top results
        maximum = max(top_results.values())
        same_similarity = {x:top_results[x] for x in top_results if top_results[x] == maximum}
        print(master[intent]['answers'][rand.choice(list(same_similarity.keys()))])


def qna_smltk_intent(data, silent=True):
    """Matches provided data to either small talk (smltk) or question & answer (qna). Uses trained classifier.

    Args:
        data (str): User query to be matched.
        silent (bool, optional): Show chosen intent on screen. Defaults to True.

    Returns:
        str: Predicted intents:
        - "smltk"
        - "qna"
    """
    if isinstance(data, str):
        data = [data]
    data_counts = countVect.transform(data)
    data_tfidf = tfidf_transformer.transform(data_counts)
    intent = classifier.predict(data_tfidf)[0]
    if not silent:
        print(intent)
    return intent


###################
### Name Intent ###
###################


def name_intent(query, verbose=False):
    """Checks if string 'query' matches the naming intent (recall/change). Gives appropriate answer to the user if it does.

    Args:
        query (str): User query.
        verbose (bool): Show probability that 'query' matches recall/change intents. Defaults to False.

    Returns:
        bool: 
        - True if intent was name (recall/change).
        - False if intent was not name (recall/change).
    """
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
        'my',
        'it'
    ]


    f_query = format(query, sw=True, stem=True)

    binary_recall = [False for _ in range(len(recall_vocabulary))]
    binary_change = [False for _ in range(len(change_vocabulary))]

    for i, word in enumerate(format(recall_vocabulary, sw=True, stem=True)):
        if word[0] in f_query and not binary_recall[i]:
            binary_recall[i] = True

    for i, word in enumerate(format(change_vocabulary, sw=True, stem=True)):
        if word[0] in f_query and not binary_change[i]:
            binary_change[i] = True

    if verbose:
        print('recall => [' + str(binary_recall.count(True)/len(binary_recall)) + ']')
        print('change => [' + str(binary_change.count(True)/len(binary_change)) + ']')

    result = ""

    if binary_recall.count(True) > binary_change.count(True) and binary_recall.count(True)/len(binary_recall) >= RECALL_THRESHOLD:
        result = "recall"
    elif binary_change.count(True) > binary_recall.count(True) and binary_change.count(True)/len(binary_change) >= CHANGE_THRESHOLD:
        result = [word for word in format(query) if word not in change_vocabulary]
        if not result: # no new name was found even though change name intent was detected
            result = "change_default"

    global USERNAME

    if result:
        if result == "recall":
            print("You said I should call you " + USERNAME + ". Anything else?")
        elif result == "change_default":
            default_answer("change")
        else:
            name = result[0].capitalize() if isinstance(result, list) and len(result)==1 else " ".join([x.capitalize() for x in result])
            print("I'll call you " + name + " from now on.")
            USERNAME = name
        return True

    return False


###############
### Helpers ###
###############


def cos_sim(q, d):
    """Computes the cosine similarity between a query and a document.

    Args:
        q ([int]): tf OR tfidf matrix of user query
        d ([int]): tf OR tfidf matrix of a document from smltk/qna datasets

    Returns:
        float: Similarity between provided query and document
    """
    x = dot(q, d)
    y = linalg.norm(q)*linalg.norm(d)
    if y == 0:
        return y
    return x/y

def format(data, sw=False, stem=False):
    """Remove punctuation & stopwords, tokenize, flatten uppercase, lemmatize given documents

    Args:
        data ([str]): List of strings to be formatted.
        sw (bool, optional): Keeps stop words if true. Defaults to False.
        stem (bool, optional): Stems rather than lemmatizing. Defaults to False.

    Returns:
        [[str]] OR [str]: List of lists of strings; each inner list contains a formatted document.
        OR list of strings if resulting list of lists has a length of 1, i.e. only one document was passed in 'data'
    """
    if isinstance(data, str):
        data = [data]

    tokenizer = nltk.RegexpTokenizer(r'\w+') # use tokenizer that removes punctuation
    nlp = spacy.load('en_core_web_sm') # spaCy english model    
    stemmer = SnowballStemmer("english")
    a = []

    for doc in data:
        if re.search('\'', doc):
            doc = contractions.fix(doc) # remove cases with apostrophe (e.g. "I'm", "it's" etc.)
        t = tokenizer.tokenize(doc) # tokenize, remove punctuation

        if stem:
            a.append([stemmer.stem(x) for x in t if x not in stopwords.words('english') or sw])
        else:
            # lemmatize, remove stopwords and flatten uppercase
            # if stop words are kept, keep pronouns as well instead of replacing them with -PRON- (spaCy)
            a.append([word.lemma_.lower() if word.lemma_ != '-PRON-' else word.lower_ for word in nlp(" ".join(t)) if not word.is_stop or sw])
    return a[0] if len(data) == 1 else a
    
def vocab(data):
    """Creates a vocabulary using the formatted 'data' passed in.

    Args:
        data ([[str]]): List of lists of strings; each inner list contains a formatted document.

    Returns:
        [str]: Vocabulary of provided data.
    """
    vocabulary = []
    for document in data:
        for term in document:
            if term not in vocabulary:
                vocabulary.append(term)
    return vocabulary

def bow(data, vocabulary):
    """Creates a simple bag-of-words model using the provided data and vocabulary. No weighting is applied (term-frequency matrix).

    Args:
        data ([[str]]): List of lists of strings; each inner list contains a formatted document.
        vocabulary ([str]): Vocabulary of provided data.

    Returns:
        [numpy array]: Bag of words modelled in a list of numpy arrays. Each numpy array contains a document's term-frequency matrix.
    """
    if not data:
        return np.zeros(len(vocabulary))

    if isinstance(data[0], str):
        data = [data]

    bow = []
    for i, doc in enumerate(data):
        bow.append(np.zeros(len(vocabulary)))
        for term in doc:
            try:
                index = vocabulary.index(term)
            except ValueError:
                if DEBUG:
                    print("\033[1;31;40mValueError:\n \'" + term + "\'\033[0m not in vocabulary. Term will be ignored.")
                continue
            bow[i][index] += 1
    return bow

def apply_tfidf(formatted_data, vocabulary):
    """Applies term-frequency, inverse-document-frequency weighting on provided formatted dataset using 
    custom functions to create a Bag-of-Words model that is then passed through a default TfidfTransformer.

    Args:
        formatted_data ([str]): List of formatted strings that will be matched against provided vocabulary
        vocabulary ([str]): smltk/qna vocabulary. 

    Returns:
        ndarray array: Matrix of tfidf values from formatted_data.
    """
    counts = bow(formatted_data, vocabulary)
    return TfidfTransformer(sublinear_tf=True).fit_transform(counts)


#######################
### Dataset Loading ###
#######################


def load_datasets():
    """Loads saved datasets from previous tests. If selected dataset doesn't exist, it is generated, saved and returned.

    Returns:
        dictionary of dictionaries: Master dictionary containing important datasets.
    """
    ###### QNA Dataset ######

    # special categories:
    ### all => 3 meaningful columns stacked on top of each other
    ### combined => concatenation of 2 or more meaningful categories
    ###             number of documents stays the same but length of each document increases
    ###             QD => question + document
    ###             Have to manual change which columns you want to use in combined_categories list

    qna_dataset = pd.read_csv('./data/qna/dataset.csv')
    qna_category = "question"
    extra = "" # any extra specifications; i.e. sw = stop words are not removed, stem = dataset is stemmed instead of lemmatised, sw_stem = combination of previous two
    qna_path = f'./objects/qna/{qna_category}_{extra}'
    all_categories = qna_dataset['question'].tolist() + qna_dataset['answer'].tolist() + qna_dataset['document'].tolist()
    all_categories_answers = qna_dataset['answer'].tolist() + qna_dataset['answer'].tolist() + qna_dataset['answer'].tolist()
    combined_categories = [x+y for x,y in zip(qna_dataset['question'], qna_dataset['document'])]

    if qna_category == 'all':
        qna_unparsed = all_categories
    elif re.search('combined', qna_category):
        qna_unparsed = combined_categories
    else:
        qna_unparsed = qna_dataset[qna_category]

    qna_sw = False
    qna_stem = False

    if re.search('sw', extra):
        qna_sw = True
    if re.search('stem', extra):
        qna_stem = True

    try:
        qna_parsed = load(f'{qna_path}.parsed.joblib')
    except:
        print(f"Parsing {qna_category}_{extra} dataset...")
        qna_parsed = format(qna_unparsed, stem=qna_stem, sw=qna_sw)
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
        qna_bag = bow(qna_parsed, qna_vocabulary)
        dump(qna_bag, f'{qna_path}.bow.joblib')

    try:
        qna_tfidf = load(f'{qna_path}.tfidf.joblib')
    except:
        print("Applying tfidf weighting to bag of words...")
        qna_tfidf = apply_tfidf(qna_parsed, qna_vocabulary)
        dump(qna_tfidf, f'{qna_path}.tfidf.joblib')


    ###### Small Talk Dataset ######

    personality = PERSONALITY # choose from ['witty', 'caring', 'enthusiastic', 'friendly', 'professional']
    variation = "dl" # choose from ['ds', 'dl'] referring to what is considered a document (ds -> section, dl -> line/question), go to definition of read_small_talk_dataset(...) for more info
    extra = "stem_sw" # same as above
    questions, answers = read_small_talk_dataset(personality, variation)
    smltk_path = f'./objects/small_talk/{personality}_{variation}_{extra}'

    smltk_sw = False
    smltk_stem = False

    if re.search('sw', extra):
        smltk_sw = True
    if re.search('stem', extra):
        smltk_stem = True

    try:
        smltk_parsed = load(f'{smltk_path}.parsed.joblib')
    except:
        print(f"Parsing {personality}_{variation}_{extra} small talk dataset...")
        smltk_parsed = format(questions, stem=smltk_stem, sw=smltk_sw)
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
        smltk_bag = bow(smltk_parsed, smltk_vocabulary)
        dump(smltk_bag, f'{smltk_path}.bow.joblib')

    try:
        smltk_tfidf = load(f'{smltk_path}.tfidf.joblib')
    except:
        print("Applying tfidf weighting to bag of words...")
        smltk_tfidf = apply_tfidf(smltk_parsed, smltk_vocabulary)
        dump(smltk_tfidf, f'{smltk_path}.tfidf.joblib')

    return {
        "smltk":{
            "unparsed": questions, # list of original questions as seen in csv file
            "parsed": smltk_parsed, # list of formatted questions (go to definition of function format(...) for more info)
            "vocabulary": smltk_vocabulary, # vocabulary of parsed dataset
            "bag": smltk_bag, # bag of words model for parsed datset
            "tfidf": smltk_tfidf, # bag of words with tfidf weighting for parsed dataset
            "answers": answers, # answers to questions
            "sw": smltk_sw, # boolean value; does parsed data contain stop words?
            "stem": smltk_stem # boolean value; was parsed data stemmed?
        },
        "qna":{
            "unparsed": qna_unparsed, #same as above
            "parsed": qna_parsed,
            "vocabulary": qna_vocabulary,
            "bag": qna_bag,
            "tfidf": qna_tfidf,
            "answers": qna_dataset['answer'] if qna_category != 'all' else all_categories_answers,
            "sw": qna_sw,
            "stem": qna_stem
        }
    }

def read_small_talk_dataset(personality, variation="ds"):
    """Reads chosen small talk dataset into a manageable list.

    Args:
        personality (str): Chosen personality from provided small talk datasets.
        Appropriate values:
        - check data/small_talk
        - each file is a different personality: "chitchat_{personality}.qna"
        variation (str, optional): Choose what a document would be.
        Appropriate values:
        - "ds" -> document-section. Each section of a small talk dataset is considered as one document. A section is all the questions corresponding to one answer.
        - "dl" -> document-line. Each question of a small talk dataset is considered as one document.
        Defaults to "ds".

    Returns:
        tuple([str], [str]): Two lists of strings of same length. First one contains questions, second contains answers.
        Indeces between lists are same, i.e. question[1928] is answered by answer[1928].
    """
    data_path = './data/small_talk/chitchat_{}.qna'
    filename = data_path.format(personality)

    try:
        f = open(filename, 'r')
    except FileNotFoundError:
        print(f'{filename} not found. Defaulting to witty personality.')
        f = open(data_path.format('witty'), 'r')
    
    questions = {x:[] for x in range(100)}
    answers = []

    index = 0

    # read all questions from file
    # question starts with '-'
    # answer is between two lines that start with '`'
    # save questions in a dictionary where the keys correspond to the index of the answer in the answers list
    # answers contain empty entries which are removed later
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
        # each answer appears as many times as the documents (lines) it corresponds to
        answers = [a for a, q in zip(answers, questions.values()) for _ in range(len(q))]
        questions = [question for section in questions.values() for question in section]
    elif variation == "ds":
        #all questions of a section are joined together separated by one white space
        questions = [" ".join(questions[i]) for i in range(len(answers))] 

    return questions, answers


##################
### Classifier ###
##################


def init_classifier(SEED=rand.randint(1,1000000), print_evaluation=False):
    """Initialise a classifier that classifies query strings as either smltk or qna.

    Args:
        SEED (int, optional): Provide specific seed for classifier. Defaults to rand.randint(1,1000000).
        print_evaluation (bool, optional): Choose whether to print the evaluation of the classifier or not.
        Evaluation includes confusion matrix, accuracy score and f1 score. Defaults to False.

    Returns:
        Object: trained classifier
    """
    qna_csv = pd.read_csv('./data/qna/dataset.csv')
    qna_dataset = [item for column in qna_csv if column != 'questionID' for item in qna_csv[column]] 

    smltk_dataset, _ = read_small_talk_dataset(PERSONALITY, 'dl')

    labels = ["smltk" for _ in range(len(smltk_dataset))]
    labels += ["qna" for _ in range(len(qna_dataset))]

    combined_dataset = smltk_dataset + qna_dataset

    # remember:
    # x = data
    # y = labels
    x_train, x_test, y_train, y_test = train_test_split(combined_dataset, labels, stratify=labels, test_size=0.01, random_state=SEED)

    x_train_counts = countVect.fit_transform(x_train)
    x_train_tf = tfidf_transformer.fit_transform(x_train_counts)

    classifier = LogisticRegression(random_state=SEED).fit(x_train_tf, y_train)

    if print_evaluation:
        x_test_counts = countVect.transform(x_test)
        x_test_tf = tfidf_transformer.transform(x_test_counts)

        predicted = classifier.predict(x_test_tf)

        print(confusion_matrix(y_test, predicted))
        print(accuracy_score(y_test, predicted))
        print(f1_score(y_test, predicted, pos_label='smltk'))
        print(f'SEED: {SEED}')

    return classifier




#####################
### Miscellaneous ###
#####################


def choose_personality():
    global PERSONALITY
    personalities = {
        'p': 'professional',
        'c': 'caring',
        'f': 'friendly',
        'e': 'enthusiastic',
        'w': 'witty',
    }

    print('Which personality would you like?')
    for key, item in personalities.items():
        print(f'\t{key} - {item}')
    x = prompt("").lower()

    if x not in personalities.keys():
        print("\033[1;31;40m\'" + x + "\'\033[0m not recognised as a valid personality. Defaulting to professional.")
        PERSONALITY = personalities['p']
    else:
        PERSONALITY = personalities[x]

def run_test_queries(typos=False):
    """Sequentially runs the provided test queries and prints the results on screen.
    Run by prompting the bot with the phrases:
    - "run test queries"
    - "run test queries with typos"

    Args:
        typos (bool, optional): Test with typos. Defaults to False.
    """
    if typos:
        f = open('./data/qna/test_queries_typos.txt', 'r')
    else:
        f = open('./data/qna/test_queries.txt', 'r')

    global USERNAME

    for line in f:
        print("\n~~~~~ " + line[:-1] + " ~~~~~\n")
        query = line[line.find(':')+2:-1]

        name = name_intent(query, verbose=False)
        if name:
            continue
        intent = qna_smltk_intent(query, silent=True)
        qna_smltk_answer(query, intent, verbose=True, tfidf=True)

    f.close()

def default_answer(intent=""):
    if intent == "qna":
        print("I don't have any such information in my database.")
    elif intent == "smltk":
        print("I am not sure what you mean.")
    elif intent == "change":
        print("Sorry, didn't get that. Can you repeat what you would me to call you?")
    else:
        print("Sorry, I can't answer that right now.")


choose_personality()
countVect = CountVectorizer(stop_words=stopwords.words('english')) 
tfidf_transformer = TfidfTransformer(use_idf=True, sublinear_tf=True)
classifier = init_classifier(SEED=78054, print_evaluation=False)
master = load_datasets()


start()


