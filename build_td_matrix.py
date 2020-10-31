from joblib import dump, load
from format_document import format
import pandas as pd

data = pd.read_csv('./dataset.csv')
DOCUMENT = 'Question'
M_PATH = f'./matrices/{DOCUMENT}'

formatted_data = []

try:
    formatted_data = load(f'{M_PATH}.matrix.joblib')
except:
    formatted_data = format(data[DOCUMENT])
    dump(formatted_data, f'{M_PATH}.matrix.joblib')

vocabulary = []
for document in formatted_data:
    for term in document:
        if term not in vocabulary:
            vocabulary.append(term)

