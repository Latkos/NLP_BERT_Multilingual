import os
from unicodedata import name

import pandas as pd
import numpy as np

import nltk
from datasets import Dataset, load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
import torch


nltk.download('punkt')
dict_labels = {
    'O': 0,
    "B-ENTITIY_1": 1,
    "I-ENTITIY_1": 2,
    "B-ENTITIY_2": 3,
    "I-ENTITIY_2": 4
}
label_names = list(dict_labels.keys())


def read_tsv_files(tsv_files=['en-small_corpora_train.tsv']):
    """Read the list of tsv files, merge it and shuffle the rows

    Args:
        tsv_files (list): list of tsv files

    Returns:
        pandas.DataFrame: Merged and Shuffled Data frame
    """

    df_list = list()
    for tsv_file in tsv_files:
        df_list.append(pd.read_csv(tsv_file, sep='\t', header=0))
    df_all = pd.concat(df_list).reset_index()
    df_ranodmize_rows = df_all.sample(frac=1)
    return df_ranodmize_rows


def create_tokens(raw_text):
    """
    Clears a sentence from tags (<e1>, </e1>, <e2>, </e2>)
    and builds a token list.

    Args:
        raw_text (str): Raw text from data frame row

    Returns:
        list: list of tokens
    """

    new_text = raw_text.replace(
        '<e1>', ' ').replace(
            '</e1>', ' ').replace(
                '<e2>', ' ').replace(
                    '</e2>', ' ').replace(
                        '.', ' . ')
    tokens = nltk.word_tokenize(new_text)
    return tokens


def create_ner_tags(raw_text):
    """Create ner tags for raw text

    Args:
        raw_text (str): Raw text from data frame row

    Returns:
        list: list of ner tags
    """
    ner_tags_list = []
    new_text = raw_text.replace(
        '.', ' . ').replace(
            '<e1>', ' StartEnity1 ').replace(
                '</e1>', ' StopEnity1 ').replace(
                    '<e2>', ' StartEnity2 ').replace(
                        '</e2>', ' StopEnity2 ')

    text_tokens = nltk.word_tokenize(new_text)
    flag = 0
    for token in text_tokens:
        if token in ['StartEnity1', 'StopEnity1', 'StartEnity2', 'StopEnity2']:
            if token == 'StartEnity1':
                flag = 1
            elif token == 'StopEnity1':
                flag = 0
            elif token == 'StartEnity2':
                flag = 3
            elif token == 'StopEnity2':
                flag = 0
        elif flag == 0:
            ner_tags_list.append(dict_labels.get('O'))
        elif flag == 1:
            ner_tags_list.append(dict_labels.get("B-ENTITIY_1"))
            flag = 2
        elif flag == 2:
            ner_tags_list.append(dict_labels.get("I-ENTITIY_1"))
        elif flag == 3:
            ner_tags_list.append(dict_labels.get("B-ENTITIY_2"))
            flag = 4
        elif flag == 4:
            ner_tags_list.append(dict_labels.get("I-ENTITIY_2"))

    return ner_tags_list


def add_tokens_ner_tags(data_frame):
    """Add column with list of tokens and list of ner tags.

    Args:
        data_frame (pandas.DataFrame): Raw Data frame

    Returns:
        pandas.DataFrame: Data frame with extra column 'tokens' and 'ner_tags'
    """
    data_frame['tokens'] = data_frame.apply(
        lambda row: create_tokens(row.text), axis=1)
    data_frame['ner_tags'] = data_frame.apply(
        lambda row: create_ner_tags(row.text), axis=1)
    return data_frame


if __name__ == '__main__':
    a = read_tsv_files(tsv_files=['en-small_corpora_train.tsv'])
    print(a.head())
    a2 = add_tokens_ner_tags(a)
    print(a2.head())
