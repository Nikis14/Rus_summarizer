"""
This code is for comparing time between models
"""
import pandas as pd
from src.Rus_summarizer import TextRankSummarizer, PageRankSummarizer, KMeansSummarizer, MixedSummarizer
import time

#necessary for working with EnSBERT
yandex_api = '<your api for yandex.translate>'


def check_time(summarizer, title):
    """Prints time necessary for model to generate summary
    Works with Time_Benchmark.csv

    Parameters
    ----------
    summarizer : <not specified>
        Summarizer that has method "generate_summary(param)"
    title: str
        The title to print before printing time
    """
    print(title+':')
    for text, sent_len in benchmark_data:
        start_time = time.perf_counter()
        summarizer.generate_summary(text)
        end_time = time.perf_counter()
        print(sent_len, ':', end_time - start_time)
    print('============================', )


#Reading Benchmark
path_to_read = '../data/Benchmarks/'
df = pd.read_csv(path_to_read + 'Time_Benchmark.csv', sep='|')
texts = list(df['text'])
lens = list(df['sentences_len'])
benchmark_data = list(zip(texts, lens))

#-----------------------------------------------------------
print('KMeans', end='\n\n')

summarizer = KMeansSummarizer.TfIdfKMeansSummarizer()
check_time(summarizer, 'tf-idf')
del summarizer

summarizer = KMeansSummarizer.MlSBERTKMeansSummarizer()
check_time(summarizer, 'multilingual')
del summarizer

summarizer = KMeansSummarizer.FastTextKMeansSummarizer()
check_time(summarizer, 'fasttext')
del summarizer

summarizer = KMeansSummarizer.RuBERTKMeansSummarizer()
check_time(summarizer, 'rubert')
del summarizer

summarizer = KMeansSummarizer.RuSBERTKMeansSummarizer()
check_time(summarizer, 'rusbert')
del summarizer

summarizer = KMeansSummarizer.EnSBERTKMeansSummarizer(yandex_api)
check_time(summarizer, 'ensbert')
del summarizer

#-----------------------------------------------------------
print('\n\nPageRank', end='\n\n')

summarizer = PageRankSummarizer.SimplePageRankSummarizer()
check_time(summarizer, 'simple pagerank:')
del summarizer

summarizer = PageRankSummarizer.MlSBERTPageRankSummarizer()
check_time(summarizer, 'multilingual:')
del summarizer

summarizer = PageRankSummarizer.FastTextPageRankSummarizer()
check_time(summarizer, 'fasttext')
del summarizer


summarizer = PageRankSummarizer.RuBERTPageRankSummarizer()
check_time(summarizer, 'rubert')
del summarizer


summarizer = PageRankSummarizer.RuSBERTPageRankSummarizer()
check_time(summarizer, 'rusbert')
del summarizer

summarizer = PageRankSummarizer.EnSBERTPageRankSummarizer(yandex_api)
check_time(summarizer, 'ensbert')
del summarizer

summarizer = TextRankSummarizer.TextRankSummarizer()
check_time(summarizer, 'TextRank:')
del summarizer

#-----------------------------------------------------------
print('\n\nMixed Summarizer:', end='\n\n')

summarizer = MixedSummarizer.MixedSummarizer()
check_time(summarizer, 'Mixed:')
del summarizer

