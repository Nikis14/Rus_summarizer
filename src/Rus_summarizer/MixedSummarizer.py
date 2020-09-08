from src.Rus_summarizer.download_tools import download_MlSBERT
import pandas as pd
from nltk.corpus import stopwords
stop_words = stopwords.words('english') + stopwords.words('russian')
import src.preprocessing_tools as pt
from itertools import combinations
from nltk import word_tokenize
import networkx as nx
import math
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class MixedSummarizer():
    """
    Class used to represent Mixed summarizer (TextRank + MlSBERT_KMeans)

    Attributes
    ----------
    sent_textrank_num: int
        base number of sentences by textrank
    model: SBERT
        MultlingualSBERT model

    Methods
    -------

    tokenize(text, min_sent_len=pt.min_sent_len)
        tokenizes text into sentences
    encode_sentences(self, sentences, additional_args)
        extracts features from sentences
    cluster_sentences(sentence_embeddings, max_count=10)
        clusterizes sentences
    _get_result_summary(clusters, sentences, sentence_embeddings) - private
        creates summary by sentences and clusters
    similarity(s1, s2)
        calculates similarity between 2 sentences
    get_sentences_textrank(sentences)
        get the most significant sentences by textrank
    get_sentences_kmeans(sentences, base_sent_count)
        get sentences of summary by kmeans
    generate_summary(text, base_sent_cout=7, min_sent_len=pt.min_sent_len)
        generates summary of text
    """
    sent_textrank_num = 3

    def __init__(self):
        super().__init__()
        download_MlSBERT("../models")
        self.model = SentenceTransformer("../models/multilingual_SBERT")

    def tokenize(self, text, min_sent_len=pt.min_sent_len):
        return pt.tokenize(text, min_sent_len)

    def encode_sentences(self, sentences, additional_args=None):
        return self.model.encode(sentences)

    def cluster_sentences(self, sentence_embeddings, max_count=10):
        """cluster sentences"""
        max_score = -1
        best_clustering = None
        for count_sentences in range(4, max_count + 1):
            clustering = KMeans(n_clusters=count_sentences, random_state=0).fit(sentence_embeddings)
            cur_score = silhouette_score(sentence_embeddings, clustering.labels_)
            if cur_score > max_score:
                best_clustering = clustering
                max_score = cur_score
        return best_clustering

    def _get_result_summary(self, clusters, sentences, sentence_embeddings):
        """form summary from clusters"""
        sent_dist = clusters.transform(sentence_embeddings).min(axis=1)
        df = pd.DataFrame({'sentences': sentences, 'dist': sent_dist, 'label': clusters.labels_})
        grouped_df = df.groupby('label')
        sentence_ids = grouped_df.apply(lambda df: df.dist.idxmin())
        return list(sentence_ids)

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

    def get_sentences_textrank(self, sentences):
        """
        get the most significant sentences by textrank

        Parameters
        ----------
        sentences : list
            list of sentences
        """
        words = []
        for sent in sentences:
            cur_words = set()
            for word in word_tokenize(sent, language='russian'):
                cur_words.add(pt.lemmatize_word(word, True))
            words.append(cur_words)

        pairs = combinations(range(len(sentences)), 2)
        scores = [(i, j, self.similarity(words[i], words[j])) for i, j in pairs]
        scores = filter(lambda x: x[2], scores)

        g = nx.Graph()
        g.add_weighted_edges_from(scores)
        pr = nx.pagerank(g)

        return sorted((i for i, s in enumerate(sentences) if i in pr),
                      key=lambda x: pr[x], reverse=True)

    def get_sentences_kmeans(self, sentences, base_sent_count):
        sentence_embeddings = self.encode_sentences(sentences)
        max_count = min(base_sent_count, len(sentences) - 1)
        clusters_kmeans_adjusted = self.cluster_sentences(sentence_embeddings,
                                                          max_count=max(max_count, len(sentences) // 30))
        if clusters_kmeans_adjusted is None:
            return []
        return self._get_result_summary(clusters_kmeans_adjusted, sentences, sentence_embeddings)

    def generate_summary(self, text, base_sent_count=7, min_sent_len=pt.min_sent_len, additional_args=None):
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
        """
        sentences = self.tokenize(text, min_sent_len)
        if len(sentences) <= 4:
            return text
        sentence_ids_kmeans = self.get_sentences_kmeans(sentences, base_sent_count-1)
        tr_num = max(self.sent_textrank_num, base_sent_count - len(sentence_ids_kmeans))
        sentence_ids_tr = self.get_sentences_textrank(sentences)

        sentence_ids = set(sentence_ids_kmeans + sentence_ids_tr[:tr_num])
        sentence_ids = sorted(sentence_ids)
        res = []
        for sent_num in sentence_ids:
            res.append(sentences[sent_num])
        return ' '.join(res)





