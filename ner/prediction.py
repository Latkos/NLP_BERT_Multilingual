from transformers import (
    AutoTokenizer,
    BertForTokenClassification,
    pipeline
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
    """Load Model from saved model file

    Args:
        model_name (string): Model name

    Returns:
        huggingface.ForTokenClassification: Token classification
    """
    model_path = NERConfig.MODEL_SAVE_PATH + model_name
    model = BertForTokenClassification.from_pretrained(
        model_path, num_labels=len(NERConfig.LABEL_NAMES))
    return model


def get_entities_sentence(list_entity_groups):
    """
    Postprocessing for prediction result.

    Args:
        list_entity_groups (list): List of predicted group for words
        e.g:
            {
                'entity_group': 'LABEL_0',
                'score': 0.99553,
                'word': 'developed by',
                'start': 190,
                'end': 202
            },
            {
                'entity_group': 'LABEL_1',
                'score': 0.7543062,
                'word': 'TI',
                'start': 203,
                'end': 205
            },
            {
                'entity_group': 'LABEL_0',
                'score': 0.9995715,
                'word': '.',
                'start': 205,
                'end': 206
            }]

    Returns:
        dict:
            'ENTITY_1': string coresponded with entity 1,
            'ENTITY_2': string coresponded with entity 2,
            'TEXT': input text with <e1>, </e1>, <e2>, </e2> tags set
    """
    words_list = []
    entity_1_words = []
    entity_2_words = []
    flag = 0

    for group in list_entity_groups:
        if group.get('entity_group') == 'LABEL_1':
            if flag == 0:
                words_list.append('<e1>')
            word = group.get('word')
            words_list.append(word)
            entity_1_words.append(word)
            flag = 1
        elif group.get('entity_group') == 'LABEL_2':
            word = group.get('word')
            words_list.append(word)
            entity_1_words.append(word)
        elif group.get('entity_group') == 'LABEL_3':
            if flag == 0:
                words_list.append('<e2>')
            word = group.get('word')
            words_list.append(word)
            entity_2_words.append(word)
            flag = 2
        elif group.get('entity_group') == 'LABEL_4':
            word = group.get('word')
            words_list.append(word)
            entity_2_words.append(word)
        elif group.get('entity_group') == 'LABEL_0':
            word = group.get('word')
            if flag == 1:
                words_list.append('</e1>')
                flag = 0
            elif flag == 2:
                words_list.append('</e2>')
                flag = 0
            words_list.append(word)

    sentence = ' '.join(words_list)
    entity_1 = ' '.join(entity_1_words)
    entity_2 = ' '.join(entity_2_words)
    return {
        'ENTITY_1': entity_1,
        'ENTITY_2': entity_2,
        'TEXT': sentence
    }


def prediction(model_name, sentences):
    """
    Prediction a entity 1 and entity 2 for sentence or list sentences 

    Args:
        model_name (string): Saved model name
        sentences (list or string): text to predict entity 1 and entity 2

    Returns:
        list: list of dict for each sentence with entity_1, entity_2 and
        input sentence with <e1>, </e1>, <e2>, </e2> tags set. 
    """

    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name)
    result = []

    token_classifier = pipeline(
        "token-classification", model=model, tokenizer=tokenizer,
        aggregation_strategy="simple"
    )

    groups = token_classifier(sentences)

    if isinstance(groups[0], dict):
        result.append(get_entities_sentence(groups))
    else:
        for i in groups:
            result.append(get_entities_sentence(i))

    return result


if __name__ == '__main__':
# EXAMPLES FROM en-small_corpa test

# 21657	Cameron	Terminator	no_relation	<e1>Cameron</e1> first gained recognition for directing The <e2>Terminator</e2> (1984).	en    
# 32831	Saskatoon Sanatorium	sanatorium	has-type	The <e1>Saskatoon Sanatorium</e1> was a tuberculosis <e2>sanatorium</e2> established in 1925 by the Saskatchewan Anti-Tuberculosis League as the second Sanatorium in the province in Wellington Park south or the Holiday Park neighborhood of Saskatoon, Saskatchewan, Canada.	en
# 16057	Caspian tiger	270-295 cm	has-length	The <e1>Caspian tiger</e1> ranked among the largest cats that ever existed. Males had a body length of <e2>270-295 cm</e2> (106-116 in) and weighed 170-240 kg (370-530 lb); females measured 240-260 cm (94-102 in) in head-to-body and weighed 85-135 kg (187-298 lb).	en
# 5169	Ashton, West Virginia	25503	post-code	Ashton has a post office with ZIP code <e2>25503</e2>. Geological Survey Geographic Names Information System: <e1>Ashton, West Virginia</e1> ZIP Code Lookup Archived June 14, 2011, at the Wayback Machine Kenny, Hamill (1945).	en
# 10410	Kirrawee High School	Kirrawee, New South Wales, Australia	is-where	<e1>Kirrawee High School</e1> is a comprehensive co-educational high school located in <e2>Kirrawee, New South Wales, Australia</e2>, adjacent to the Royal National Park.	en
# 458	Ambara	Sen Prakash	movie-has-director	<e1>Ambara</e1> (Kannada: ಅಂಬರ) is a 2013 Indian Kannada language romance film written and directed by <e2>Sen Prakash</e2>.	en

# 744	Mani di fata	Steno	movie-has-director	<e1>Mani di fata</e1> (Fairy hands) is a 1983 Italian comedy film directed by <e2>Steno</e2>.	en
# 13746	Female southern elephant seals	400 to 900 kg	has-weight	On average <e1>Female southern elephant seals</e1> weigh <e2>400 to 900 kg</e2> (880 to 1,980 lb).	en
# 21023	Penumbra	Penumbra: Overture	no_relation	Built upon this engine they made <e1>Penumbra</e1>, a tech demo to display the engine's capabilities, which later evolved into their first game, <e2>Penumbra: Overture</e2>.	en

    model_name = 'en-small_corpora'
    # sentece = "Ambara (Kannada: ಅಂಬರ) is a 2013 Indian Kannada language romance film written and directed by Sen Prakash."
    # result = prediction(model_name, sentece)
    # print(result, "\n\n")
    model_name = 'en-small_corpora'
    senteces = [
        "Mani di fata (Fairy hands) is a 1983 Italian comedy film directed by Steno.",
        "On average Female southern elephant seals weigh 400 to 900 kg (880 to 1,980 lb).",
        "Built upon this engine they made Penumbra, a tech demo to display the engine's capabilities, which later evolved into their first game, Penumbra: Overture."
    ]

    result = prediction(model_name, senteces)
    print(result)
