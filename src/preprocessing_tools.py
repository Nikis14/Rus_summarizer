"""
Functions for text preprocessing
"""
import re
import os
import pymorphy2
from rusenttokenize import ru_sent_tokenize
import logging
logging.disable(logging.CRITICAL)
from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english') + stopwords.words('russian'))


morph = pymorphy2.MorphAnalyzer() #pymorphy2 lemmatizer
min_sent_len = 20 #mininmum length of a sentence to include into consideration


def lemmatize_word(word, remove_stopwords=False):
    """lemmatize word

    Parameters
    ----------
    word : str
        word to lemmatize
    remove_stopwords: bool
        flag whether remove stop words
    """
    word = word.lower()
    if not word[0].isalpha():
        return ''
    if len(re.findall('[a-z0-9]', word)):
        return word
    parsed = morph.parse(word)[0]
    lemma = parsed.normal_form
    if remove_stopwords and word in stop_words:
        return ''
    return lemma


def lemmatize_text(text, remove_stopwords=False):
    """lemmatize all words in text

    Parameters
    ----------
    text : str
        text to lemmatize
    remove_stopwords: bool
        flag whether remove stop words
    """
    res = []
    for word in word_tokenize(text, language='russian'):
        res.append(lemmatize_word(word, remove_stopwords))
    return ' '.join(res)


def preprocess_text(text):
    """preprocess text

    Parameters
    ----------
    text : str
        text to preprocess
    """
    text = text.replace('\n', ' ').replace('\r', '')
    text = re.sub('\.+', '..', text)
    text = text.replace('«', '"').replace('»', '"')
    text = re.sub('[\.\?\!\;]"', '"', text)
    text = re.sub('[\.\?\!]\;"', ';', text)
    text = re.sub(' +', ' ', text)
    text = re.sub('\t+', ' ', text)
    return text


def tokenize_with_lemmatization(text, min_len=min_sent_len, remove_stopwords=False):
    """tokenize and lemmatize text

    Parameters
    ----------
    text : str
        text to process
    min_len : int
        minimum length of sentence to remain
    remove_stopwords: bool
        flag whether remove stop words
    """
    text = preprocess_text(text)
    sentences = ru_sent_tokenize(text)
    res = []
    cleaned_sentences = []
    for sentence in sentences:
      sentence = sentence.strip()
      if len(sentence) >= min_len:
        res_sent = []
        for word in word_tokenize(sentence):
          word = lemmatize_word(word, remove_stopwords)
          res_sent.append(word)
        cleaned_sentences.append(' '.join(res_sent))
        res.append(sentence)
    return res, cleaned_sentences


def tokenize(text, min_len=25):
    """tokenize text

    Parameters
    ----------
    text : str
        text to process
    min_len : int
        minimum length of sentence to remain
    """
    text = preprocess_text(text)
    sentences = ru_sent_tokenize(text)
    res = []
    for sentence in sentences:
      sentence = sentence.strip()
      if len(sentence) >= min_len:
        res.append(sentence)
    return res


def preprocess_path(path):
    # add / to path if necessary
    if path == '':
        return path
    if path[:-1] != '/':
        path += '/'
    return path


def create_dir(directory):
    #create directory
    if not os.path.exists(directory):
        os.makedirs(directory)

