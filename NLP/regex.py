import numpy as np
from scrapy import Selector
import requests

url = 'https://elpais.com/economia/2020-04-02/la-seguridad-social-pierde-833000-afiliados-y-el-paro-sube-en-mas-de' \
      '-300000-personas-en-el-peor-mes-para-el-empleo.html'

html = requests.get(url=url).content
selection = Selector(text=html)
raw_data = selection.extract()
print(raw_data)