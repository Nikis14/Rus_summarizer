"""
Example of MixedSummarizer work
"""

import pandas as pd
from src.Rus_summarizer import MixedSummarizer

summarizer = MixedSummarizer.MixedSummarizer()
df = pd.read_csv('../data/analyzed_data/Simple_PageRank/Tech.csv', sep='|', encoding='utf8')
text = df['text'][14]
print(summarizer.generate_summary(text))
