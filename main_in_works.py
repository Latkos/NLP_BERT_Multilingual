import click

# from ner import training as ner_training
# import pandas as pd

# from relations_model import train_model, evaluate_model


@click.group()
def bert_cli():
    """
        Command Line Interpreter for multilingual m-bert model
        for named entity recognition (NER) and relations extraction (RE)
    """
    pass


@bert_cli.command(
help="""Train M-BERT model for named entity recognition (NER) and
relations extraction (RE) with train file test file and config file
""", short_help="Train M-BERT model")
@click.option('--config', default='./config/train.cfg', type=str, help='path to train config')
@click.option('--model_name', default=1, type=str, help='model name')
@click.argument('train_file', nargs=1, required=True, type=str)
@click.argument('test_file', nargs=1, required=True, type=str)
def train(train_file, test_file, config, model_name):
    print(train_file)
    print(test_file)
    print(config)
    print(model_name)

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
