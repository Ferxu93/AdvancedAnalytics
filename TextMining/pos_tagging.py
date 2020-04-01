import nltk
nltk.download('brown')

from nltk.corpus import brown

brown_news_tagged = brown.tagged_sents(categories='news', tagset='universal')
brown_news_words = brown.tagged_words(categories='news',  tagset='universal')

brown_train = brown_news_tagged[100:]
brown_test = brown_news_tagged[:100]

from nltk.tag import untag
test_sent = untag(brown_test[0])
print("Tagged: ", brown_test[0])
print()
print("Untagged: ", test_sent)

# A default tagger assigns the same tag to all words
from nltk import DefaultTagger
default_tagger = DefaultTagger('NOUN')
default_tagger.tag('This is a test'.split())