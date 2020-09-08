"""
Generate summary for all texts in dataframe by path
"""
import os
import pandas as pd
from src.preprocessing_tools import preprocess_path, min_sent_len, create_dir

#generate summary for all texts in dataframe by path
def analyze_dfs(
        path_to_read,
        path_to_save,
        summarizer,
        base_sent_cout=8,
        min_sent_len=min_sent_len,
        versions=None,
        new_col='generated',
        text_col='text',
        df_sep='|'
):
  path_to_read = preprocess_path(path_to_read)
  files = os.listdir(path_to_read)
  if versions is None:
    path_to_save = preprocess_path(path_to_save)
    create_dir(path_to_save)
    for file in files:
      df = pd.read_csv(path_to_read + file, sep=df_sep, encoding='utf8')
      df[new_col] = df[text_col].apply(lambda text: summarizer.generate_summary(text, base_sent_cout, min_sent_len))
      df.to_csv(path_to_save + file, sep=df_sep, encoding='utf8')

  else:
    for version in versions:
      new_path = path_to_save + '_' + version[0]
      new_path = preprocess_path(new_path)
      create_dir(new_path)
      for file in files:
        df = pd.read_csv(path_to_read + file, sep=df_sep, encoding='utf8')
        df[new_col] = df[text_col].apply(lambda text: summarizer.generate_summary(text, base_sent_cout, min_sent_len, version[1]))
        df.to_csv(new_path + file, sep=df_sep, encoding='utf8')


