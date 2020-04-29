import numpy as np
import pandas as pd
import re


desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 30)
pd.options.display.float_format = '{:.3f}'.format

train_df = pd.read_csv('/Users/fer/Downloads/train.csv')
test_df = pd.read_csv('/Users/fer/Downloads/test.csv')
sample_df = pd.read_csv('/Users/fer/Downloads/sample_submission.csv')

print(train_df.head(20))
print(test_df.head(20))
print(sample_df.head(20))

''' EDA :: Basic statistics  and type analysis '''

print('-------TRAIN--------')
print('Train types\n', train_df.dtypes)
print('Train info\n', train_df.info())
print('Train description\n', train_df.describe())
print('Train shape\n', train_df.shape)

nas_keyword = sum(pd.isnull(train_df['keyword']))
print('nas keyword', nas_keyword)
nas_location = sum((pd.isnull(train_df['location'])))
print('nas location', nas_location)

print('---------------------')

print('------TEST-----')
print('test types\n', test_df.dtypes)
print('test info\n', test_df.info())
print('test description\n', test_df.describe())
print('test shape\n', test_df.shape)

nas_keyword = sum(pd.isnull(test_df['keyword']))
print('nas keyword',nas_keyword)
nas_location = sum((pd.isnull(test_df['location'])))
print('nas location', nas_location)

print('-----------------')

''' Subset keyword and location columns :: for != to NaN'''
train_k = train_df[train_df.keyword.notnull()].reset_index()
print('Train set with all observations containing any valid keyword\n', train_k)
train_l = train_df[train_df.location.notnull()].reset_index()
print('Train set with all observations containing any valid location\n', train_l)
print(train_l.info())
train_kl = train_df[(train_df.keyword.notnull()) & (train_df.location.notnull())].reset_index()
test_dfx = test_df[(test_df.keyword.notnull()) & (test_df.location.notnull())].reset_index()
print(train_kl)

''' ReGex '''
print(train_kl.text)
hashtags = re.findall(r'#.+?\s', string=str(train_kl.text))
hashtags_t = re.findall(r'#.+?\s', string=str(test_df.text))
print(hashtags)
hashtags = [re.findall(r'#.+?\s', item) for item in train_df.text]
hashtags_t = [re.findall(r'#.+?\s', item) for item in test_df.text]
print(hashtags)
print('length of hashtags: ', len(hashtags))

hashtags = []
for item in train_df.text:
    loquesea = re.findall(r'#.+?\s', item)
    hashtags.append(loquesea)

users = [re.findall(r'@.+?\s', item) for item in train_df.text]
users_t = [re.findall(r'@.+?\s', item) for item in test_df.text]
print(users)
print('lenght of users: ', len(users))

# add hashtags and users to the main dataframe:
train_df['hashtags'] = [item[0] if type(item) == 'list' else item for item in hashtags]
train_df['users'] = [item[0] if type(item) == 'list' else item for item in users]
test_df['hashtags'] = [item[0] if type(item) == 'list' else item for item in hashtags_t]
test_df['users'] = [item[0] if type(item) == 'list' else item for item in users_t]
print(train_df)

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

sentences = [sent_tokenize(sentence) for sentence in train_df.text]
print(sentences)
print(len(sentences))

tokens_per_sent = [word_tokenize(str(t).lower()) for t in sentences]
print(tokens_per_sent)
print(len(tokens_per_sent))

tokens = []
for sentence in tokens_per_sent:
    logic = []
    tokens.append(logic)
    for word in sentence:
        if word.isalpha():
            logic.append(word)

train_df['tokens'] = tokens
print(train_df)

from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from gensim import models

import itertools

dictionary = Dictionary(train_df.tokens)
corpus = dictionary.doc2bow(list(itertools.chain(*train_df.tokens))) # bow
print(corpus)
print(len(corpus))

tfidf = TfidfModel([corpus])
tfidf_weights = tfidf[corpus]

doc_weight = []
for tweet in train_df.tokens:
    for word in tweet:
        dictionary.token2id.get(word)

from nltk import pos_tag
import nltk

nltk.download('averaged_perceptron_tagger')

tagged_tokens = [pos_tag(tag) for tag in train_df.tokens]
print(tagged_tokens)
print(len(tagged_tokens))

train_df['tagged'] = tagged_tokens
print(train_df)

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()
X_train_df = ohe.fit(pd.Series(train_df.dropna()))

''' Splitting between Train and test sets '''
model = 0
if model == 1:

    Splitting_activation = 1
    if Splitting_activation == 1:

        X = train_df # predictive variables
        y = train_df['target'] # 1 malignant and 0 benignant

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=235)

    ''' Model selection '''

    svm_model = svm.SVC(kernel='linear') # instantiate
    svm_model.fit(X_train, y_train)
    y_pred_svm_model= svm_model.predict(X_test)
    print('This is my linear model:\n',y_pred_svm_model)

    print('Accuracy: {:.2f}'.format(metrics.accuracy_score(y_test, y_pred_svm_model)))
    print('Precision: {:.2f}'.format(metrics.precision_score(y_test, y_pred_svm_model)))
    print('Recall: {:.2f}'.format(metrics.recall_score(y_test, y_pred_svm_model)))

    svm_model = svm.SVC(kernel='poly')
    svm_model.fit(X_train, y_train)
    y_pred_svm_model= svm_model.predict(X_test)
    print('This is my Polinomial model:\n',y_pred_svm_model)

    print('Accuracy: {:.2f}'.format(metrics.accuracy_score(y_test, y_pred_svm_model)))
    print('Precision: {:.2f}'.format(metrics.precision_score(y_test, y_pred_svm_model)))
    print('Recall: {:.2f}'.format(metrics.recall_score(y_test, y_pred_svm_model)))

    svm_model = svm.SVC(kernel='rbf')
    svm_model.fit(X_train, y_train)
    y_pred_svm_model= svm_model.predict(X_test)
    print('This is my Radial Basis model:\n',y_pred_svm_model)

    print('Accuracy: {:.2f}'.format(metrics.accuracy_score(y_test, y_pred_svm_model)))
    print('Precision: {:.2f}'.format(metrics.precision_score(y_test, y_pred_svm_model)))
    print('Recall: {:.2f}'.format(metrics.recall_score(y_test, y_pred_svm_model)))
