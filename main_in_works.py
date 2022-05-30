import click

from ner import training as ner_training
from ner import prediction as ner_prediction
from config_parser import get_ner_training_args

# import pandas as pd

# from relations_model import train_model, evaluate_model


@click.group()
def bert_cli():
    """
Command Line Interpreter for multilingual m-bert model\
for named entity recognition (NER) and relations extraction (RE)

Example:

- python main_in_works.py train ./data/en-small_corpora_train.tsv \
./data/en-small_corpora_test.tsv --config ./config/base_config.yaml \
--model_name en-small_corpora

- python main_in_works.py predict "Kyrgyz International Airlines was an airline based in Kyrgyzstan." \
--model_name en-small_corpora
"""
    pass


@bert_cli.command(
    help="""Train M-BERT model for named entity recognition (NER) and
relations extraction (RE) with train file test file and config file""",
    short_help="Train M-BERT model",
)
@click.option("--config", default="./config/train.cfg", type=str, help="path to train config")
@click.option("--model_name", default="en-small-corpora", type=str, help="model name")
@click.argument("train_file", nargs=1, required=True, type=str)
@click.argument("test_file", nargs=1, required=True, type=str)
def train(train_file, test_file, config, model_name):
    args = get_ner_training_args(config)
    ner_test_result = ner_training.train_model(
        train_tsv_file=train_file, test_tsv_file=test_file, model_name=model_name, training_arguments=args
    )
    print(f"Test result after training: {ner_test_result}")


@bert_cli.command(
    help="""Predict entity_1, entity_2 and relation between entities with \
pretrained M-BERT model for inputed text""",
    short_help="Predict entity_1, entity_2, relation",
)
@click.argument("text", nargs=-1, required=True, type=str)
@click.option("--model_name", nargs=1, default="en-small-corpora", type=str, help="model name")
def predict(model_name, text):
    print(model_name, "\n\n\n")
    sentence = " ".join(text)
    print(sentence, "\n\n\n")

    prediction_result = ner_prediction.prediction(model_name=model_name, sentences=sentence)
    print(prediction_result)

    # ner_test_result = ner_training.train_model(
    #     train_tsv_file=train_file, test_tsv_file=test_file,
    #     model_name=model_name, training_arguments=args)
    # print(f"Test result after training: {ner_test_result}")


# def main():
#     # TODO loading and parsing arguments, remove hardcoding
#     train_filename = "en-small_corpora_train.tsv"
#     test_filename = "data_tinkering/en-small_corpora_test.tsv"
#     model_name = "../second-relations"
#     subset_percentage = 0.5
#     train_df = pd.read_csv(train_filename, sep="\t")
#     train_df = train_df.sample(frac=subset_percentage)
#     train_model(train_df, model_name)
#     test_df = pd.read_csv(test_filename, sep="\t")
#     evaluate_model(test_df, model_name)


if __name__ == "__main__":
    bert_cli()
