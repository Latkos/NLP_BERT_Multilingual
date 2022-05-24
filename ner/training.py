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
LABEL_NAMES = list(dict_labels.keys())
MODEL_NAME = 'bert-base-multilingual-cased'
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
DATA_COLLATOR = DataCollatorForTokenClassification(TOKENIZER)
METRIC = load_metric("seqeval")


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


def create_dataset(data_frame):
    """Convert tokens and ner_tags columns from data frame to Dataset.

    Args:
        data_frame (pandas.DataFrame):

    Returns:
        Dataset: Dataset object
    """

    new_data_frame = data_frame[['tokens', 'ner_tags']]
    return Dataset.from_pandas(new_data_frame)


def tokenize_adjust_labels(all_samples_per_split):
    """
    Adjust labes. For each sample, we need to get the values for input_ids,
    token_type_ids and attention_mask as well as adjust the labels.

    Args:
        all_samples_per_split (Dataset.row): Dataset row

    Returns:
        Dataset: Dataset with tokenized samples.
    """
    tokenized_samples = TOKENIZER.batch_encode_plus(
        all_samples_per_split["tokens"], is_split_into_words=True)
    # tokenized_samples is not a datasets object so
    # this alone won't work with Trainer API, hence map is used
    # so the new keys [input_ids, labels (after adjustment)]
    # can be added to the datasets dict for each train test validation split
    total_adjusted_labels = []
    print(len(tokenized_samples["input_ids"]))
    for k in range(0, len(tokenized_samples["input_ids"])):
        prev_wid = -1
        word_ids_list = tokenized_samples.word_ids(batch_index=k)
        existing_label_ids = all_samples_per_split["ner_tags"][k]
        i = -1
        adjusted_label_ids = []

        for wid in word_ids_list:
            if (wid is None):
                adjusted_label_ids.append(-100)
            elif (wid != prev_wid):
                i = i + 1
                adjusted_label_ids.append(existing_label_ids[i])
                prev_wid = wid
            else:
                label_name = LABEL_NAMES[existing_label_ids[i]]
                adjusted_label_ids.append(existing_label_ids[i])

        total_adjusted_labels.append(adjusted_label_ids)
    tokenized_samples["labels"] = total_adjusted_labels
    return tokenized_samples


def preprocess_dataset(tsv_files=['en-small_corpora_train.tsv']):
    """
    Preprocess the Dataset.
    1. Read all data frames,
    2. Add tokens nad ner tags
    3. Convert to dataset
    4. Tokenize and adjust labels
    5. return dataset used for training or test model

    Args:
        tsv_files (list): list of tsv files

    Returns:
        Dataset: Dataset with tokenized samples.
    """
    df1 = read_tsv_files(tsv_files=['en-small_corpora_train.tsv'])
    print(df1.head())
    df2 = add_tokens_ner_tags(df1)
    print(df2.head())
    d1 = create_dataset(df2)
    print(d1)
    d2 = d1.map(tokenize_adjust_labels, batched=True)
    return d2


def compute_metrics(p):
    """Calculate metrics for val dataset

    Args:
        p (tuple): prediction

    Returns:
        dict: metrics result
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [LABEL_NAMES[p] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    true_labels = [
        [LABEL_NAMES[l] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]

    results = METRIC.compute(
        predictions=true_predictions, references=true_labels)

    flattened_results = {
        "overall_precision": results["overall_precision"],
        "overall_recall": results["overall_recall"],
        "overall_f1": results["overall_f1"],
        "overall_accuracy": results["overall_accuracy"],
    }
    for k in results.keys():
        if (k not in flattened_results.keys()):
            flattened_results[k + "_f1"] = results[k]["f1"]

    return flattened_results


if __name__ == '__main__':
    a = preprocess_dataset(tsv_files=['en-small_corpora_train.tsv'])
    print(a)
