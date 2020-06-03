"""
This code contains functions for evaluation of dataframe in which every text contains:
1) Annotation (column "annotation")
2) Generated summary (column "generated")
3) Key words (column "key_words")
Evaluation metrics: ROUGE-1, ROUGE-2, ROUGE_L, key words%
"""
from preprocessing_tools import lemmatize_text, preprocess_path
import pymorphy2
import os
import pandas as pd
import numpy as np
from rouge import Rouge


morph = pymorphy2.MorphAnalyzer() #lemmatizer for russian
rouge = Rouge() #evaluating object


def estimate_key_words(generated_text, key_words_str):
    """evaluate key words %

    Parameters
    ----------
    generated_text : str
        generated summary
    key_words_str: str
        key words separated by ','
    """
    key_words = lemmatize_text(key_words_str, True).split(' ')
    generated_text_check = set(generated_text.split(' '))
    res = 0
    for key_word in key_words:
        if key_word in generated_text_check:
            res += 1
    return res/len(key_words)


def process_df(df):
    """evaluate df by ROUGE and key words

    Parameters
    ----------
    df : dataframe
        dataframe with generated summaries
    """
    df['generated_lemma'] = df['generated'].apply(lambda text: lemmatize_text(text, False))
    df['annotation_lemma'] = df['annotation'].apply(lambda text: lemmatize_text(text, False))
    rouge_1 = []
    rouge_2 = []
    rouge_l = []
    key_word_est = []
    all_rouges = [rouge_1, rouge_2, rouge_l]
    order_names = ['rouge-1', 'rouge-2', 'rouge-l']
    order_scores = ['f', 'p', 'r']
    for i, row in df.iterrows():
        scores = rouge.get_scores(row['generated_lemma'], row['annotation_lemma'])[0]
        rouge_1.append(list(scores['rouge-1'].values()))
        rouge_2.append(list(scores['rouge-2'].values()))
        rouge_l.append(list(scores['rouge-l'].values()))

        key_word_est.append(estimate_key_words(row['generated_lemma'], row['key_words']))

    for i in range(len(order_names)):
        cur_array = np.array(all_rouges[i]).T
        for j in range(len(order_scores)):
            df[order_names[i] + '_' + order_scores[j]] = cur_array[j]

    df['key_word_part'] = key_word_est
    return df


def estimate_dfs(path_to_data):
    """evaluate dfs by path

    Parameters
    ----------
    path_to_data : str
        path to dataframes
    """
    path_to_data = preprocess_path(path_to_data)
    folders = os.listdir(path_to_data)
    for folder in folders:
        if folder != 'Mixed_ML_TR':
            continue
        folder += '/'
        files = os.listdir(path_to_data + folder)
        for file in files:
            path_to_file = path_to_data + folder + file
            df = pd.read_csv(path_to_file, sep='|', encoding='utf8')
            df = process_df(df)
            df.to_csv(path_to_file, sep='|', encoding='utf8')
        print(folder, 'done')

