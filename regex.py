import pandas as pd
import numpy as np
from scrapy import Selector
import requests

url = 'https://www.abc.es/espana/abci-pleno-congreso-sobre-crisis-coronavirus-directo-202003251459_directo.html'
html = requests.get(url=url).content
selection = Selector(text=html)
raw_data = selection.extract()

''' REGEX => Regular Expressions '''
# ++++++++++++++++++ pre-regex +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
news_title = 'Coronavirus en España, última hora en directo: los fallecidos en todo el mundo superan los 20.000'
rest_of_title = '          muertos ...'

# length of the title:
print('This sentence has {:.0f} words'.format(len(news_title)))

# number of words:
splitted_sentence = news_title.split() # split in words a sentence
number_of_words = len(splitted_sentence)
print('The news has {:.0f}'.format(number_of_words))

# join the rest of the title to the title:
complete_news_title = news_title + ' ' + rest_of_title
print(complete_news_title)

# pick up the last word (in the title, the number of deaths ...):
splitted_sentence = complete_news_title.split()
print('list: ', splitted_sentence)
print('Number of deaths is: ', splitted_sentence[-3])
print(splitted_sentence[0], splitted_sentence[2], splitted_sentence[8])
print(complete_news_title.find('spa')) # returns the position in which the search appears

complete2_news_title = news_title + ' ' + rest_of_title.strip()
print(complete2_news_title)

print('++++++++++++++++++++++++++++++++++++++++++++\n\n')
print(raw_data.count('Iglesias'))

fake_title = complete2_news_title
print(fake_title.replace('España', 'Reino Unido'))ng