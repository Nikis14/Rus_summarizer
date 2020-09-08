"""
This code is for generating summaries by all summarizers
"""
from src.summarizers import TextRankSummarizer, PageRankSummarizer, KMeansSummarizer, MixedSummarizer
from src.processing_texts import analyze_dfs

path_to_read = '../data/main_data/'
path_to_save = '../data/analyzed_data/'

print('TF-IDF_KMeans')
summarizer = KMeansSummarizer.TfIdfKMeansSummarizer()
analyze_dfs(path_to_read, path_to_save + 'TF-IDF_KMeans', summarizer)
del summarizer

print('MultiLingual_KMeans')
summarizer = KMeansSummarizer.MlSBERTKMeansSummarizer()
analyze_dfs(path_to_read, path_to_save + 'MultiLingual_KMeans', summarizer)
del summarizer

print('FastText_KMeans')
summarizer = KMeansSummarizer.FastTextKMeansSummarizer()
analyze_dfs(path_to_read, path_to_save + 'FastText_KMeans', summarizer, versions=[('Clean', True), ('Raw', False)])
del summarizer
print('Simple_PageRank')
summarizer = PageRankSummarizer.SimplePageRankSummarizer()
analyze_dfs(path_to_read, path_to_save + 'Simple_PageRank', summarizer, base_sent_cout=6)
del summarizer


print('TF-IDF_PageRank')
summarizer = PageRankSummarizer.TfIdfPageRankSummarizer()
analyze_dfs(path_to_read, path_to_save + 'TF-IDF_PageRank', summarizer, base_sent_cout=6)
del summarizer


print('Multilingual_PageRank')
summarizer = PageRankSummarizer.MlSBERTPageRankSummarizer()
analyze_dfs(path_to_read, path_to_save + 'Multilingual_PageRank', summarizer, base_sent_cout=5)
del summarizer

print('FastText_PageRank')
summarizer = PageRankSummarizer.FastTextPageRankSummarizer()
analyze_dfs(path_to_read, path_to_save + 'FastText_PageRank', summarizer, base_sent_cout=5, versions=[('Clean', True), ('Raw', False)])
del summarizer

print('RuBERT_KMeans')
summarizer = KMeansSummarizer.RuBERTKMeansSummarizer()
analyze_dfs(path_to_read, path_to_save + 'RuBERT_KMeans', summarizer, versions=[('Without_ST', False)])
del summarizer

print('RuSBERT_KMeans')
summarizer = KMeansSummarizer.RuSBERTKMeansSummarizer()
analyze_dfs(path_to_read, path_to_save + 'RuSBERT_KMeans', summarizer, versions=[('With_ST', True), ('Without_ST', False)])
del summarizer

print('RuBERT_PageRank')
summarizer = PageRankSummarizer.RuBERTPageRankSummarizer()
analyze_dfs(path_to_read, path_to_save + 'RuBERT_PageRank', summarizer, base_sent_cout=5, versions=[('With_ST', True), ('Without_ST', False)])
del summarizer

print('RuSBERT_PageRank')
summarizer = PageRankSummarizer.RuSBERTPageRankSummarizer()
analyze_dfs(path_to_read, path_to_save + 'RuSBERT_PageRank', summarizer, base_sent_cout=5, versions=[('With_ST', True), ('Without_ST', False)])
del summarizer

print('TextRank')
summarizer = TextRankSummarizer.TextRankSummarizer()
analyze_dfs(path_to_read, path_to_save + 'TextRank', summarizer, base_sent_cout=6)
del summarizer


print('Mixed')
summarizer = MixedSummarizer.MixedSummarizer()
analyze_dfs(path_to_read, path_to_save + 'Mixed_ML_TR', summarizer, base_sent_cout=8)
del summarizer