# Алгоритм TextRank в исходном виде
from src import preprocessing_tools as pt
from itertools import combinations
from nltk import word_tokenize
import networkx as nx
import math


class TextRankSummarizer():
    """
    Class used to represent TextRank summarizer

    Attributes
    ----------

    Methods
    -------
    similarity(s1, s2)
        calculates similarity between 2 sentences
    encode_sentences(self, sentences, additional_args=None)
        extracts features from sentences
    text_rank(text, min_sent_len)
        calculates significance of sentences
    generate_summary(text, base_sent_cout=6, min_sent_len=pt.min_sent_len)
        generates summary of text
    """
    def __init__(self):
        pass

    def similarity(self, s1, s2):
        """
        calculates similarity between 2 sentences

        Parameters
        ----------
        s1 : set
            set of words in sentence 1
        s2 : set
            set of words in sentence 2
        """
        if not len(s1) or not len(s2):
            return 0.0
        return len(s1.intersection(s2)) / (math.log(len(s1) + 1) + math.log(len(s2) + 1))

    def text_rank(self, text, min_sent_len):
        """
        calculates significance of sentences

        Parameters
        ----------
        text : str
            set of words in sentence 1
        min_sent_len : int
            set of words in sentence 2
        """
        sentences, clean_sentences = pt.tokenize_with_lemmatization(text, min_sent_len, True)
        if len(sentences) < 2:
            s = sentences[0]
            return [(1, 0, s)]

        words = [set(word_tokenize(sent)) for sent in (clean_sentences)]

        pairs = combinations(range(len(sentences)), 2)
        scores = [(i, j, self.similarity(words[i], words[j])) for i, j in pairs]
        scores = filter(lambda x: x[2], scores)

        g = nx.Graph()
        g.add_weighted_edges_from(scores)
        pr = nx.pagerank(g)

        return sorted(((i, pr[i], s) for i, s in enumerate(sentences) if i in pr),
                      key=lambda x: pr[x[0]], reverse=True)

    def generate_summary(self, text, base_sent_count=6, min_sent_len=pt.min_sent_len, add_special_tokens=None):
        """
        generate summary of the text

        Parameters
        ----------
        text : str
            input text
        base_sent_cout: int
            count of sentences in summary
        min_sent_len: int
            minimum length of sentence to remain
        add_special_tokens: ?
            fictitious parameter
        """
        tr = self.text_rank(text, min_sent_len)
        top_n = sorted(tr[:base_sent_count])  # Сортировка первых n предложений по их порядку в тексте
        return ' '.join(x[2] for x in top_n)  # Соединяем предложения