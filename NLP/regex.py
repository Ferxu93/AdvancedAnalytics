import numpy as np
from scrapy import Selector
import requests
import re

url = 'https://elpais.com/economia/2020-04-02/la-seguridad-social-pierde-833000-afiliados-y-el-paro-sube-en-mas-de' \
      '-300000-personas-en-el-peor-mes-para-el-empleo.html'

html = requests.get(url=url).content
selection = Selector(text=html)
raw_data = selection.extract()
print(raw_data)

''' Basic ReGex '''

'''
Metacharacters:
 => (1)
+ => (1, inf) :: greedy matching
* => (0, inf) :: greedy matching
how to stop greed (called also non-greedy matching):
      ? (begins with ?)
. = anything => (1)
      .+ => anything till the end

Regex commands:
\w => one letter (NOT WORD!!!) :: case INsensitive  
\W => everything excepts a letter (i.e.: 8, 12, ., /, $, ...)
\d => one digit
\D => everything excepts a digit

Operators:
| => OR 
[] => OR (within a range)

Special characters (metacharacters) as normal text:
\\, \+, \*, \.

'''
print('\n\n+++++++++++++++++++++++++++++ REGEX CASES +++++++++++++++++++++++++++++++++++++++++')
# list with all times that ""Seguridad Social, seguridad social or Seguridad social"" appears:

Seguridad_social = re.findall(r'Seguridad Social', raw_data)
print(Seguridad_social)
print(len(Seguridad_social))

ministerioA = re.findall(r'Ministerio|ministerio', raw_data) # OR explicit
print(ministerioA)

ministerioB = re.findall(r'\winisterio', raw_data) # direct
print(ministerioB)

ministerioC = re.findall(r'[mM]inisterio', raw_data)
print(ministerioC)

# list of all url's (websites) within the raw data:
urls = re.findall(r'(https*:.+?)">', raw_data)
print(urls)
print(len(urls))

# extract all names with its first surname:
names = re.findall(r'[A-Z]\w+\s[A-Z]\w+\s', raw_data)
#print(names)

text = 'We have so many workload. I had 12 meetings during the week and I expect to have ' \
       'more than 20 during the next week. I hope to go for holidays sooner than after. ' \
       '#stopcoronavirus @Jaime. @Nacho, help me finish the fuck*** homework!!!'

# change many to much, as many does not apply:
text_1 = re.sub(pattern=r'many', repl='much', string=text)
print(text_1)

# create two objects: hastags and users and fill them with the correspondent data
hashtags = re.findall(r'(#.+?)\s', text)
users = re.findall(r'(@[A-Z].*?)\W', text)
print(hashtags)
print('Users:')
print(users)

''' Advanced ReGex 
========================0
?= => positive lookahead  => in front-of the regex
?! => negative lookahead
?<= => positive lookbehind
?<! => negative lookbehind

'''

# extract the adverb before meetings:
number_of_meetings = re.findall(r'\w+(?=\sworkload)', text) # positive lookahead
print('result:', number_of_meetings)

# extract with a negative lookahead the "estado de animo" distinct of rainy:
case = 'happy if sunny, sad if rainy, crying if snowing'
feelings = re.findall(r'\w+\s(?!rainy)', case) # negative lookahead
print(feelings)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
''' Tokenization '''
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
import os

import nltk
try:
    if not os.path.exists('/Users/fer/nltk_data/tokenizers/punkt'):
        nltk.download('punkt')
except ValueError:
    print('Value Error')

words = word_tokenize(text=text, language='english')
print(words)
print(len(words))

sentences = sent_tokenize(text=text, language='english')
print(sentences)
print(len(sentences))
