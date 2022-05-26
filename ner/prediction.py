from unittest import result
import pandas as pd
import numpy as np
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    BertForTokenClassification
)

from ner_config import NERConfig


def load_tokenizer(model_name):
    """Load tokenizer from saved model file

    Args:
        model_name (string): Model name

    Returns:
        huggingface.Tokenizer: Tokenizer
    """
    model_path = NERConfig.MODEL_SAVE_PATH + model_name
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer


def load_model(model_name):
    """Load Mode from saved model file

    Args:
        model_name (string): Model name

    Returns:
        huggingface.TokenClassification: Token classification
    """
    model_path = NERConfig.MODEL_SAVE_PATH + model_name
    model = BertForTokenClassification.from_pretrained(
        model_path, num_labels=len(NERConfig.LABEL_NAMES))
    return model


def convert_subwords_list(subwords_list):
    words_list = list()
    tmp = ''
    for i in subwords_list:
        if i.startswith('##'):
            tmp += i.replace('##', '')
        else:
            if tmp == '':
                tmp = i
            else:
                words_list.append(tmp)
                tmp = i
    words_list.append(tmp)
    return words_list


def prediction(model_name, sentence):
    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name)

    tokens = tokenizer(sentence)
    input_ids = tokens['input_ids']
    predictions = model.forward(
        input_ids=torch.tensor(input_ids).unsqueeze(0),
        attention_mask=torch.tensor(tokens['attention_mask']).unsqueeze(0))
    label_id = torch.argmax(predictions.logits.squeeze(), axis=1)
    list_label_names = [NERConfig.LABEL_NAMES[i] for i in label_id]

    if len(list_label_names) != len(input_ids):
        raise Exception("Can not add tags <e1>, <e2> to sentence")

    word_list = []
    entity_1 = []
    entity_2 = []
    flag = 0
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    for indx, input_id in enumerate(input_ids):
        if (input_id == cls_id) or (input_id == sep_id):
            continue
        if (list_label_names[indx] == "B-ENTITIY_1"):
            if flag == 0:
                word_list.append('<e1>')
            word = tokenizer.convert_ids_to_tokens(input_id)
            word_list.append(word)
            entity_1.append(word)
            flag = 1
        elif (list_label_names[indx] == "I-ENTITIY_1"):
            word = tokenizer.convert_ids_to_tokens(input_id)
            word_list.append(word)
            entity_1.append(word)
        elif (list_label_names[indx] == "B-ENTITIY_2"):
            if flag == 0:
                word_list.append('<e2>')
            word = tokenizer.convert_ids_to_tokens(input_id)
            word_list.append(word)
            entity_2.append(word)
            flag = 2
        elif (list_label_names[indx] == "I-ENTITIY_2"):
            word = tokenizer.convert_ids_to_tokens(input_id)
            word_list.append(word)
            entity_2.append(word)
        elif (list_label_names[indx] == "O"):
            if flag == 1:
                word = tokenizer.convert_ids_to_tokens(input_id)
                word_list.append('</e1>')
                word_list.append(word)
                flag = 0
            elif flag == 2:
                word = tokenizer.convert_ids_to_tokens(input_id)
                word_list.append('</e2>')
                word_list.append(word)
                flag = 0
            else:
                word = tokenizer.convert_ids_to_tokens(input_id)
                word_list.append(word)

    new_word_list = convert_subwords_list(word_list)
    new_entity_1 = convert_subwords_list(entity_1)
    new_entity_2 = convert_subwords_list(entity_2)
    sentence = ' '.join(new_word_list)

    return new_entity_1, new_entity_2, sentence


if __name__ == '__main__':
# 21657	Cameron	Terminator	no_relation	<e1>Cameron</e1> first gained recognition for directing The <e2>Terminator</e2> (1984).	en    
# 32831	Saskatoon Sanatorium	sanatorium	has-type	The <e1>Saskatoon Sanatorium</e1> was a tuberculosis <e2>sanatorium</e2> established in 1925 by the Saskatchewan Anti-Tuberculosis League as the second Sanatorium in the province in Wellington Park south or the Holiday Park neighborhood of Saskatoon, Saskatchewan, Canada.	en
# 16057	Caspian tiger	270-295 cm	has-length	The <e1>Caspian tiger</e1> ranked among the largest cats that ever existed. Males had a body length of <e2>270-295 cm</e2> (106-116 in) and weighed 170-240 kg (370-530 lb); females measured 240-260 cm (94-102 in) in head-to-body and weighed 85-135 kg (187-298 lb).	en
# 5169	Ashton, West Virginia	25503	post-code	Ashton has a post office with ZIP code <e2>25503</e2>. Geological Survey Geographic Names Information System: <e1>Ashton, West Virginia</e1> ZIP Code Lookup Archived June 14, 2011, at the Wayback Machine Kenny, Hamill (1945).	en
# 10410	Kirrawee High School	Kirrawee, New South Wales, Australia	is-where	<e1>Kirrawee High School</e1> is a comprehensive co-educational high school located in <e2>Kirrawee, New South Wales, Australia</e2>, adjacent to the Royal National Park.	en

    model_name = 'en-small_corpora'
    sentece = "Kirrawee High School is a comprehensive co-educational high school located in Kirrawee, New South Wales, Australia, adjacent to the Royal National Park."
    ent_1, ent_2, new_sentence = prediction(model_name, sentece)
    print(ent_1, ent_2, new_sentence)
