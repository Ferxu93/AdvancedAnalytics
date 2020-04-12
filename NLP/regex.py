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
+ => (1, inf)
* => (0, inf)

Regex commands:
\w => one letter (NOT WORD!!!) :: case INsensitive  
\W => everything excepts a letter (i.e.: 8, 12, ., /, $, ...)
\d => one digit
\D => everything excepts a digit

Operators:
| => OR 
[] => OR (within a range)

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