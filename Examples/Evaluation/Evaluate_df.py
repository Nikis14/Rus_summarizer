"""
This code is for evaluation of dataframe in which every text contains:
1) Annotation
2) Generated summary
3) Key words
Evakuation: ROUGE-1, ROUGE-2, ROUGE_L, key words%
"""

from src import Evaluate as eval

eval.estimate_dfs('../data/analyzed_data/')