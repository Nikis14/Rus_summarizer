from src import preprocessing_tools as pt
from src.summarizers.download_tools import download_MlSBERT, download_FastText
import numpy as np
import re
from nltk import word_tokenize
from nltk.cluster.util import cosine_distance
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
stop_words = stopwords.words('english') + stopwords.words('russian')
from abc import ABC, abstractmethod
import networkx as nx
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelWithLMHead
import fasttext as ft
from yandex_translate import YandexTranslate


#Abstract class for PageRankSummarizer
class AbstractPageRankSummarizer(ABC):
    """
    Abstract class used to represent summarizers based on PageRank

    Attributes
    ----------

    Methods
    -------
    tokenize(text, min_sent_len=pt.min_sent_len)
        tokenizes text into sentences
    build_similarity_matrix(sentences, additional_args)
        build matrix of sentences similarity
    generate_summary(text, base_sent_cout=5, min_sent_len=pt.min_sent_len, additional_args=None)
        generates summary of text
    """
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def tokenize(self, text, min_sent_len=pt.min_sent_len):
        """tokenize text into sentences"""
        pass

    @abstractmethod
    def build_similarity_matrix(self, sentences, additional_args):
        """build matrix of similarity between sentences"""
        pass

    def generate_summary(self, text, base_sent_count=5, min_sent_len=pt.min_sent_len, additional_args=None):
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

        # Step 1 - Read text and split it
        sentences = self.tokenize(text, min_sent_len)
        if len(sentences) <= 4:
            return text

        # Step 2 - Generate Similary Martix across sentences
        sentence_similarity_martix = self.build_similarity_matrix(sentences, additional_args)

        # Step 3 - Rank sentences in similarity martix
        sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
        scores = nx.pagerank(sentence_similarity_graph, alpha=0.6)

        # Step 4 - Sort the rank and pick top sentences
        ranked_sentence = sorted(((scores[i], s, i) for i, s in enumerate(sentences)), reverse=True)

        result_sentences = ranked_sentence[:base_sent_count]
        result_sentences = sorted(result_sentences, key=lambda x: x[2])

        summarize_text = []
        for i in range(base_sent_count):

            if result_sentences[i] != []:
                summarize_text.append(result_sentences[i][1])

        return " ".join(summarize_text)


class SimplePageRankSummarizer(AbstractPageRankSummarizer):
    """
    Class used to represent Simple_PageRank summarizer

    Attributes
    ----------
    _clean_sentences: list
        list of preprocessed sentences

    Methods
    -------
    tokenize(text, min_sent_len=pt.min_sent_len)
        tokenizes text into sentences
    sentence_similarity(sent1, sent2, remove_stopwords=False)
        calculates similarity between 2 sentences
    build_similarity_matrix(sentences, additional_args)
        builds matrix of sentences similarity
    generate_summary(text, base_sent_cout=5, min_sent_len=pt.min_sent_len, remove_stopwords=False)
        generates summary of text
    """
    def __init__(self):
        super().__init__()
        self._clean_sentences = ''

    def tokenize(self, text, min_len=pt.min_sent_len):
        sentences, self._clean_sentences = pt.tokenize_with_lemmatization(text, min_len)
        return sentences

    def sentence_similarity(self, sent1, sent2, remove_stopwords=False):
        all_words = list(set(sent1 + sent2))

        vector1 = [0] * len(all_words)
        vector2 = [0] * len(all_words)

        # build the vector for the first sentence
        for w in sent1:
            if not remove_stopwords or w not in stop_words:
                vector1[all_words.index(w)] += 1

        # build the vector for the second sentence
        for w in sent2:
            if not remove_stopwords or w not in stop_words:
                vector2[all_words.index(w)] += 1

        return 1 - cosine_distance(vector1, vector2)

    def build_similarity_matrix(self, sentences, remove_stopwords):
        sentences = self._clean_sentences
        similarity_matrix = np.zeros((len(sentences), len(sentences)))

        for idx1 in range(len(sentences)):
            for idx2 in range(len(sentences)):
                if idx1 == idx2:  # ignore if both are same sentences
                    continue
                similarity_matrix[idx1][idx2] = self.sentence_similarity(sentences[idx1], sentences[idx2], remove_stopwords)
        return np.array(similarity_matrix)

    def generate_summary(self, text, base_sent_count=5, min_sent_len=pt.min_sent_len, remove_stopwords=False):
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
        remove_stopwords: bool
            remove stopwords or not
        """
        return super().generate_summary(text, base_sent_count, min_sent_len, remove_stopwords)


class RuSBERTPageRankSummarizer(AbstractPageRankSummarizer):
    """
    Class used to represent RuSBERT_PageRank summarizer

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
    build_similarity_matrix(sentences, additional_args)
        build matrix of sentences similarity
    generate_summary(text, base_sent_cout=5, min_sent_len=pt.min_sent_len, additional_args=None)
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

    def build_similarity_matrix(self, sentences, add_special_tokens):
        sentence_embeddings = self.encode_sentences(sentences, add_special_tokens)
        pre_res = cosine_similarity(sentence_embeddings)
        return np.abs(np.ones(pre_res.shape) - pre_res)

    def generate_summary(self, text, base_sent_count=5, min_sent_len=pt.min_sent_len, add_special_tokens=True):
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
        return super().generate_summary(text, base_sent_count, min_sent_len, add_special_tokens)


class RuBERTPageRankSummarizer(AbstractPageRankSummarizer):
    """
    Class used to represent RuBERT_PageRank summarizer

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
    encode_sentences(self, sentences, add_special_tokens)
        extracts features from sentences
    build_similarity_matrix(sentences, additional_args)
        build matrix of sentences similarity
    generate_summary(text, base_sent_cout=5, min_sent_len=pt.min_sent_len, additional_args=None)
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
                input_ids = torch.tensor(
                    self.tokenizer.encode('0', add_special_tokens=add_special_tokens)).unsqueeze(0)
                embeddings = self.model(input_ids=input_ids)[0]
            embeddings = embeddings.squeeze(0)
            sentence_embeddings = torch.mean(embeddings, dim=0)
            res_embeddings.append(sentence_embeddings)
        return torch.stack(res_embeddings, dim=0).detach().numpy()

    def build_similarity_matrix(self, sentences, add_special_tokens):
        sentence_embeddings = self.encode_sentences(sentences, add_special_tokens)
        pre_res = cosine_similarity(sentence_embeddings)
        return np.abs(np.ones(pre_res.shape) - pre_res)

    def generate_summary(self, text, base_sent_count=5, min_sent_len=pt.min_sent_len, add_special_tokens=True):
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
        return super().generate_summary(text, base_sent_count, min_sent_len, add_special_tokens)


class FastTextPageRankSummarizer(AbstractPageRankSummarizer):
    """
    Class used to represent FastText_PageRank summarizer

    Attributes
    ----------
    model: ft_model
        FastText model

    Methods
    -------
    tokenize(text, min_sent_len=pt.min_sent_len)
        tokenizes text into sentences
    encode_sentences(self, sentences, add_special_tokens)
        extracts features from sentences
    build_similarity_matrix(sentences, additional_args)
        build matrix of sentences similarity
    generate_summary(text, base_sent_cout=5, min_sent_len=pt.min_sent_len, additional_args=None)
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

    def build_similarity_matrix(self, sentences, clean):
        sentence_embeddings = self.encode_sentences(sentences, clean)
        pre_res = cosine_similarity(sentence_embeddings)
        return np.abs(np.ones(pre_res.shape) - pre_res)

    def generate_summary(self, text, base_sent_count=5, min_sent_len=pt.min_sent_len, clean=True):
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
        return super().generate_summary(text, base_sent_count, min_sent_len, clean)


class MlSBERTPageRankSummarizer(AbstractPageRankSummarizer):
    """
    Class used to represent MlSBERT_PageRank summarizer

    Attributes
    ----------
    model: SBERT
        MultlingualSBERT model

    Methods
    -------
    tokenize(text, min_sent_len=pt.min_sent_len)
        tokenizes text into sentences
    build_similarity_matrix(sentences, additional_args)
        build matrix of sentences similarity
    generate_summary(text, base_sent_cout=5, min_sent_len=pt.min_sent_len)
        generates summary of text
    """
    def __init__(self):
        super().__init__()
        download_MlSBERT("../models/multilingual_SBERT")
        self.model = SentenceTransformer("../models/multilingual_SBERT")

    def tokenize(self, text, min_sent_len=pt.min_sent_len):
        return pt.tokenize(text, min_sent_len)

    def build_similarity_matrix(self, sentences, additional_args):
        sentence_embeddings = self.model.encode(sentences)
        pre_res = cosine_similarity(sentence_embeddings)
        return np.abs(np.ones(pre_res.shape) - pre_res)

    def generate_summary(self, text, base_sent_count=5, min_sent_len=pt.min_sent_len):
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
        return super().generate_summary(text, base_sent_count, min_sent_len, None)


class EnSBERTPageRankSummarizer(AbstractPageRankSummarizer):
    """
    Class used to represent EnSBERT_PageRank summarizer

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
    build_similarity_matrix(sentences, additional_args)
        build matrix of sentences similarity
    generate_summary(text, base_sent_cout=5, min_sent_len=pt.min_sent_len)
        generates summary of text
    """
    def __init__(self, yandex_translate_api_key):
        super().__init__()
        self.model = model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')
        self.translate = YandexTranslate(yandex_translate_api_key)

    def tokenize(self, text, min_sent_len=pt.min_sent_len):
        return pt.tokenize(text, min_sent_len)

    def translate_to_english(self, sentences):
        res = []
        for sentence in sentences:
            sentence = self.translate.translate(sentence, 'ru-en')['text'][0]
            sentence = re.sub('\.+', '', sentence)
            sentence = re.sub('[\.\?\!]\;"', ';', sentence)
            sentence = sentence.replace('.;', ';')
            res.append(sentence)
        return res

    def build_similarity_matrix(self, sentences, additional_args):
        sentences_english = self.translate_to_english(sentences)
        sentence_embeddings = self.model.encode(sentences_english)
        pre_res = cosine_similarity(sentence_embeddings)
        return np.abs(np.ones(pre_res.shape) - pre_res)

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
