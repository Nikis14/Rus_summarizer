import preprocessing_tools as pt
from download_tools import download_MlSBERT, download_FastText
import numpy as np
import pandas as pd
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english') + stopwords.words('russian')
from abc import ABC, abstractmethod
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelWithLMHead
import fasttext as ft
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from yandex_translate import YandexTranslate


#Abstract class for KMeansSummarizer
class AbstractKMeansSummarizer(ABC):
    """
    Abstract class used to represent summarizers based on K-Means

    Attributes
    ----------

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
    generate_summary(text, base_sent_cout=8, min_sent_len=pt.min_sent_len, additional_args=None)
        generates summary of text
    """
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def tokenize(self, text, min_sent_len=pt.min_sent_len):
        """
        tokenizes text into sentences

        Parameters
        ----------
        text : str
            word to lemmatize
        min_sent_len: int
            minimum length of sentence to remain
        """
        pass

    @abstractmethod
    def encode_sentences(self, sentences, additional_args):
        """
        builds matrix of similarity between sentences

        Parameters
        ----------
        sentences : list
            list of sentences
        additional_args: ?
            additional arguments
        """
        pass

    def cluster_sentences(self, sentence_embeddings, max_count=10):
        """
        clusterizes sentences

        Parameters
        ----------
        sentence_embeddings : numpy array
            list of sentences
        max_count: int
            max count of clustsers
        """
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
        """
        creates summary by sentences and clusters
        """
        sent_dist = clusters.transform(sentence_embeddings).min(axis=1)
        df = pd.DataFrame({'sentences': sentences, 'dist': sent_dist, 'label': clusters.labels_})
        grouped_df = df.groupby('label')
        sentence_ids = grouped_df.apply(lambda df: df.dist.idxmin())
        sentence_ids = sorted(sentence_ids)
        result_sentences = list(df.loc[sentence_ids]['sentences'])
        return ' '.join(result_sentences)

    @abstractmethod
    def generate_summary(self, text, base_sent_cout=8, min_sent_len=pt.min_sent_len, additional_args=None):
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
        additional_args: ?
            additional arguments
        """
        sentences = self.tokenize(text, min_sent_len)
        if len(sentences) <= 4:
            return text

        sentence_embeddings = self.encode_sentences(sentences, additional_args)
        max_count = min(base_sent_cout, len(sentences) -  1)
        clusters_kmeans_adjusted = self.cluster_sentences(sentence_embeddings, max_count=max(max_count, len(sentences) // 30))
        if clusters_kmeans_adjusted is None:
            return text
        return self._get_result_summary(clusters_kmeans_adjusted, sentences, sentence_embeddings)


class TfIdfKMeansSummarizer(AbstractKMeansSummarizer):
    """
    Class used to represent TF-IDF_KMMeans summarizer

    Attributes
    ----------
    _clean_sentences: list
        list of preprocessed sentences

    Methods
    -------
    tokenize(text, min_sent_len=pt.min_sent_len)
        tokenizes text into sentences
    encode_sentences(self, sentences, additional_args=None)
        extracts features from sentences
    cluster_sentences(sentence_embeddings, max_count=10)
        clusterizes sentences
    _get_result_summary(clusters, sentences, sentence_embeddings) - private
        creates summary by sentences and clusters
    generate_summary(text, base_sent_cout=8, min_sent_len=pt.min_sent_len)
        generates summary of text
    """
    def __init__(self):
        super().__init__()
        self._clean_sentences = ''

    def tokenize(self, text, min_sent_len=pt.min_sent_len):
        sentences, self._clean_sentences = pt.tokenize_with_lemmatization(text, min_sent_len)
        return sentences

    def encode_sentences(self, sentences, additional_args=None):
        count_vect = CountVectorizer()
        counts = count_vect.fit_transform(self._clean_sentences)
        tfidf_transformer = TfidfTransformer()
        return tfidf_transformer.fit_transform(counts)


    def generate_summary(self, text, base_sent_cout=8, min_sent_len=pt.min_sent_len):
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
        return super().generate_summary(text, base_sent_cout, min_sent_len, None)


class RuSBERTKMeansSummarizer(AbstractKMeansSummarizer):
    """
    Class used to represent RuSBERT_KMMeans summarizer

    Attributes
    ----------
    tokenizer: AutoTokenizer
        tokenizer for RuSBERT model
    model: AutoModelWithLMHead
        RuSBERT model

    Methods
    -------
    tokenize(text, min_sent_len=pt.min_sent_len)
        tokenizes text into sentences
    encode_sentences(self, sentences, add_special_tokens)
        extracts features from sentences
    cluster_sentences(sentence_embeddings, max_count=10)
        clusterizes sentences
    _get_result_summary(clusters, sentences, sentence_embeddings) - private
        creates summary by sentences and clusters
    generate_summary(text, base_sent_cout=8, min_sent_len=pt.min_sent_len, add_special_tokens=False)
        generates summary of text
    """
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased-sentence")
        self.model = AutoModelWithLMHead.from_pretrained("DeepPavlov/rubert-base-cased-sentence")
        self.model.eval()

    def tokenize(self, text, min_sent_len=pt.min_sent_len):
        return pt.tokenize(text, min_sent_len)

    def encode_sentences(self, sentences, add_special_tokens):
        res_embeddings = []
        for sentence in sentences:
            input_ids = torch.tensor(self.tokenizer.encode(sentence, add_special_tokens=add_special_tokens)).unsqueeze(0)
            try:
                embeddings = self.model(input_ids=input_ids)[0]
            except IndexError:
                input_ids = torch.tensor(
                    self.tokenizer.encode('0', add_special_tokens=add_special_tokens)).unsqueeze(0)
                embeddings = self.model(input_ids=input_ids)[0]
            embeddings = embeddings.squeeze(0)
            sentence_embeddings = torch.mean(embeddings, dim=0)
            res_embeddings.append(sentence_embeddings)
        return torch.stack(res_embeddings, dim=0).detach().numpy()

    def generate_summary(self, text, base_sent_cout=8, min_sent_len=pt.min_sent_len, add_special_tokens=False):
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
        add_special_tokens: bool
            whether add tokens [CLS], [SEP] before use of RuSBERT
        """
        return super().generate_summary(text, base_sent_cout, min_sent_len, add_special_tokens)


class RuBERTKMeansSummarizer(AbstractKMeansSummarizer):
    """
    Class used to represent RuBERT_KMMeans summarizer

    Attributes
    ----------
    tokenizer: AutoTokenizer
        tokenizer for RuBERT model
    model: AutoModelWithLMHead
        RuBERT model

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
    generate_summary(text, base_sent_cout=8, min_sent_len=pt.min_sent_len, add_special_tokens=False)
        generates summary of text
    """
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
        self.model = AutoModelWithLMHead.from_pretrained("DeepPavlov/rubert-base-cased")
        self.model.eval()

    def tokenize(self, text, min_len=pt.min_sent_len):
        return pt.tokenize(text, min_len)

    def encode_sentences(self, sentences, add_special_tokens):
        res_embeddings = []
        for sentence in sentences:
            input_ids = torch.tensor(self.tokenizer.encode(sentence, add_special_tokens=add_special_tokens)).unsqueeze(0)
            try:
                embeddings = self.model(input_ids=input_ids)[0]
            except IndexError:
                input_ids = torch.tensor(self.tokenizer.encode('0', add_special_tokens=add_special_tokens)).unsqueeze(0)
                embeddings = self.model(input_ids=input_ids)[0]
            embeddings = embeddings.squeeze(0)
            sentence_embeddings = torch.mean(embeddings, dim=0)
            res_embeddings.append(sentence_embeddings)
        return torch.stack(res_embeddings, dim=0).detach().numpy()

    def generate_summary(self, text, base_sent_cout=8, min_sent_len=pt.min_sent_len, add_special_tokens=False):
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
        add_special_tokens: bool
            whether add tokens [CLS], [SEP] before use of RuBERT
        """
        return super().generate_summary(text, base_sent_cout, min_sent_len, add_special_tokens)


class FastTextKMeansSummarizer(AbstractKMeansSummarizer):
    """
    Class used to represent FastText_KMMeans summarizer

    Attributes
    ----------
    model: ft_model
        FastText model

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
    generate_summary(text, base_sent_cout=8, min_sent_len=pt.min_sent_len, clean=True)
        generates summary of text
    """
    def __init__(self):
        super().__init__()
        download_FastText("../models/ft_native_300_ru_wiki_lenta_lower_case")
        self.model = ft.load_model("../models/ft_native_300_ru_wiki_lenta_lower_case/ft_native_300_ru_wiki_lenta_lower_case.bin")

    def tokenize(self, text, min_sent_len=pt.min_sent_len):
        return pt.tokenize(text, min_sent_len)

    def encode_sentences(self, sentences, clean=False):
        res_embeddings = []
        if clean:
            for sentence in sentences:
                res_sent = []
                for word in word_tokenize(sentence):
                    if word[0].isalpha():
                        res_sent.append(word.lower())
                res_embeddings.append(self.model.get_sentence_vector(' '.join(res_sent)))
        else:
            for sentence in sentences:
                res_embeddings.append(self.model.get_sentence_vector(sentence.lower()))

        return np.stack(res_embeddings, axis=0)

    def generate_summary(self, text, base_sent_cout=8, min_sent_len=pt.min_sent_len, clean=True):
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
        clean: bool
            drop punctuation or not
        """
        return super().generate_summary(text, base_sent_cout, min_sent_len, clean)


class MlSBERTKMeansSummarizer(AbstractKMeansSummarizer):
    """
    Class used to represent MlSBERT_KMMeans summarizer

    Attributes
    ----------
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
    generate_summary(text, base_sent_cout=8, min_sent_len=pt.min_sent_len)
        generates summary of text
    """
    def __init__(self):
        super().__init__()
        download_MlSBERT("../models/multilingual_SBERT")
        self.model = SentenceTransformer("../models/multilingual_SBERT")

    def tokenize(self, text, min_sent_len=pt.min_sent_len):
        return pt.tokenize(text, min_sent_len)

    def encode_sentences(self, sentences, additional_args=None):
        return self.model.encode(sentences)

    def generate_summary(self, text, base_sent_cout=8, min_sent_len=pt.min_sent_len):
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
        return super().generate_summary(text, base_sent_cout, min_sent_len, None)


class EnSBERTKMeansSummarizer(AbstractKMeansSummarizer):
    """
    Class used to represent EnSBERT_KMMeans summarizer

    Attributes
    ----------
    model: SBERT
        EnSBERT model

    Methods
    -------
    tokenize(text, min_sent_len=pt.min_sent_len)
        tokenizes text into sentences
    translate_to_english(sentences)
        translate sentences from russian to english
    encode_sentences(self, sentences, additional_args)
        extracts features from sentences
    cluster_sentences(sentence_embeddings, max_count=10)
        clusterizes sentences
    _get_result_summary(clusters, sentences, sentence_embeddings) - private
        creates summary by sentences and clusters
    generate_summary(text, base_sent_cout=8, min_sent_len=pt.min_sent_len)
        generates summary of text
    """
    def __init__(self, yandex_translate_api_key):
        super().__init__()
        self.model = model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')
        self.translate = YandexTranslate(yandex_translate_api_key)

    def tokenize(self, text, min_sent_len=pt.min_sent_len):
        return pt.tokenize(text, min_sent_len)

    def translate_to_english(self, sentences):
        """
        translate sentences to english

        Parameters
        ----------
        sentences : list
            list of sentences
        """
        res = []
        for sentence in sentences:
            sentence = self.translate.translate(sentence, 'ru-en')['text'][0]
            sentence = re.sub('\.+', '.', sentence)
            sentence = re.sub('[\.\?\!]\;"', ';', sentence)
            sentence = sentence.replace('.;', ';')
            res.append(sentence)
        return res

    def encode_sentences(self, sentences, additional_args=None):
        sentences_english = self.translate_to_english(sentences)
        return self.model.encode(sentences_english)

    def generate_summary(self, text, base_sent_cout=8, min_sent_len=pt.min_sent_len):
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
        return super().generate_summary(text, base_sent_cout, min_sent_len, None)

