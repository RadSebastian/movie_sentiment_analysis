import re 
import pandas as pd
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
stop = stopwords.words('english')
porter = PorterStemmer()
df = pd.DataFrame()
df = pd.read_csv('movie_data.csv')

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower())
    text = text.join(emoticons).replace('-', '')
    return text

def tokenizer(text):
    return text.split()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

df['review'] = df['review'].apply(preprocessor)

tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)
param_grid = [{'vect__ngram_range': [(1,1)],
                'vect__stop_words': [None],
                'vect__tokenizer': [tokenizer],
                'clf__penalty': ['l2'],
                'clf__C': [10.0]},
                {'vect__ngram_range': [(1,1)],
                'vect__stop_words': [None],
                'vect__tokenizer': [tokenizer],
                'vect__use_idf':[False],
                'vect__norm':[None],
                'clf__penalty': ['l2'],
                'clf__C': [10.0]}
            ]
lr_tfidf = Pipeline([('vect', tfidf),
                        ('clf',
                        LogisticRegression(random_state=0))])
gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                            scoring='accuracy',
                            cv=5, verbose=1,
                            n_jobs=-1)

X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

gs_lr_tfidf.fit(X_train, y_train)
#print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
print('CV accuracy: %.3f'
        % gs_lr_tfidf.best_score_)
clf = gs_lr_tfidf.best_estimator_
print('Test Accuracy: %.3f'
        % clf.score(X_test, y_test))







