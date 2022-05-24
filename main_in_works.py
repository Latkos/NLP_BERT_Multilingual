import pandas as pd

from relations_model import train_model, evaluate_model


def main():
    # TODO loading and parsing arguments, remove hardcoding
    train_filename = "en-small_corpora_train.tsv"
    test_filename = "data_tinkering/en-small_corpora_test.tsv"
    model_name = "../second-relations"
    subset_percentage = 0.5
    train_df = pd.read_csv(train_filename, sep="\t")
    train_df = train_df.sample(frac=subset_percentage)
    train_model(train_df, model_name)
    test_df = pd.read_csv(test_filename, sep="\t")
    evaluate_model(test_df, model_name)


if __name__ == "__main__":
    main()
