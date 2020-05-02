import pandas as pd
import numpy as np
from scrapy import Selector
import requests
import re
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from collections import Counter
import time
from gensim.corpora.dictionary import Dictionary

start_time = time.time()


desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_colwidth', 400)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 300)
pd.options.display.float_format = '{:.3f}'.format


url = 'https://gist.githubusercontent.com/jsdario/6d6c69398cb0c731' \
      '11e49f1218960f79/raw/8d4fc4548d437e2a7203a5aeeace5477f598827d/el_quijote.txt'

html = requests.get(url=url).content
selection = Selector(text=html)
raw_data = selection.extract()
print(raw_data)

''' 
1. Lemmatizer and stemming => root of words
2. Lowering and .isalpha() => homogeneization
3. Creating a corpus and dictionary => [(id, freq), (34, 514), ()]
4. Word2vec # 
5. Tf-idf model **+ 
6. n-grams (bigrams, trigrams, ...)
'''

''' (1) Text Analytics '''
print('Length of El Quijote: ', len(raw_data))

words = word_tokenize(raw_data, language='spanish')
print('Words of El Quijote: ', len(words))
print('Type of object: ', type(words))

# print first 20 words of your words object:
f20words = words[:20]
print(f20words)

# create an object with all the sentences in the document:
sentences = sent_tokenize(raw_data, language='spanish')
print('Sentences of El Quijote:', len(sentences))
f20sentences = sentences[:20]
print(f20sentences)

# create a dataframe with the first column with the first 20 sentences:
ElQuijote_df = pd.DataFrame(f20sentences)
print(ElQuijote_df)

# create an object JUST with the first sentence:
fsentence = sentences[:1]
print('\nFirst sentence: ', fsentence)

tokens_1st_sentence = [t for t in word_tokenize(str(fsentence).lower()) if t.isalpha()]
print(tokens_1st_sentence)

tokens_1st_sentence_nsw = [t for t in tokens_1st_sentence if t not in stopwords.words('spanish')]
print(tokens_1st_sentence_nsw)

wnl = WordNetLemmatizer() # we instantiate when we load a function/class in Caps (Mayusculas)
tokens_1st_sentence_nsw_stemmed = [wnl.lemmatize(t) for t in tokens_1st_sentence_nsw]
print(tokens_1st_sentence_nsw_stemmed)

cnt = Counter(tokens_1st_sentence_nsw_stemmed)
print(cnt.most_common(2))

def book_analytics(book, language):
      '''Params:
      Token: ...'''
      sentences = [sentence for sentence in sent_tokenize(book)] # sentences here is a token
      words = [words for words in word_tokenize(str(sentences).lower()) if words.isalpha()]
      words_nsw = [nsw for nsw in words if nsw not in stopwords.words(language)]

      wnlemmatizer = WordNetLemmatizer() # class instantiation
      words_nsw_lemm = [wnlemmatizer.lemmatize(w) for w in words_nsw]
      cnt = Counter(words_nsw_lemm)
      cnts = Counter(sentences)

      print('\n++++++++++++++++++++++++++++++ BOOK SUMMARY +++++++++++++++++++++++++++++++++')
      print('\nNumber of sentences in the book: ', len(sentences))
      print('Number of words in the book: ', len(words))
      print('Number of relevant words: ', len(words_nsw))
      print('Top 10 words: ', cnt.most_common(10))
      print('Top bottom 10 words: ', cnt.most_common()[-10:])
      print('Top 10 sentences: ', pd.DataFrame(cnts.most_common(10)))


book_analytics_execute = 0
if book_analytics_execute == 1:

      quijote = book_analytics(raw_data, language='spanish')
      print(quijote)

'''
Gensim
================
1. complex model in NLP (documents and word vectors)
1.1. Word vectors : the distance between car and motorbike its closer than car and house. (Deep learning based 
     methodology for mathematical calculus of multidimensional vectors) 
1.2. LDA: Latent Dirichlet Allocation (statistic model for topic analysis and sematic distances)

2. Collections: a list containing tuples as (id, freq) => if list is ['a', 'b'], tuple is ('a', 'b') 
* RECALL => a tuple is an immutable structure (while a list could be modified, a tuple NEVER!)
'''
# step 1: tokenization, remove stop words and lemmatizer:
sentences = [sentence for sentence in sent_tokenize(raw_data)]  # sentences here is a token
words = [words for words in word_tokenize(str(sentences).lower()) if words.isalpha()]
words_nsw = [nsw for nsw in words if nsw not in stopwords.words('spanish')]

wnlemmatizer = WordNetLemmatizer()  # class instantiation
words_nsw_lemm = [wnlemmatizer.lemmatize(w) for w in words_nsw]

# step 2: we create a dictionary and a collection (BOW = Bag-Of-Words) and a corpus:
dictionary = Dictionary([words_nsw_lemm]) # Instantiate
toboso_id = dictionary.token2id.get('toboso')
print('this is the id from a given word from the book (or any text): ', toboso_id)
toboso = dictionary.get(10192)
print(toboso)

corpus = dictionary.doc2bow(words_nsw_lemm) # this is the collection (brutally connected with the dictionary)
print(corpus)

id714 = dictionary.get(714)
print(id714)


end_time = time.time()
print('\nExecution time: {:.2f}s'.format(end_time - start_time))






















