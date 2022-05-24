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



if __name__ == '__main__':
    a = read_tsv_files()
    print(a.head())