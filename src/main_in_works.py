import pandas as pd
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    pipeline,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from relations_dataset import RelationsDataset

from logger import logger


def get_texts_and_labels(df, subset_percentage=1.0):
    texts = df["text"].tolist()
    labels = df["label"].tolist()
    map = dict([(y, x) for x, y in enumerate(sorted(set(labels)))])
    labels = [map[x] for x in labels]
    return texts, labels


def prune_prefixes_from_labels(predictions):
    predicted_labels = []
    for prediction in predictions:
        label = prediction["label"]
        label = label.replace("LABEL_", "")
        label = int(label)
        predicted_labels.append(label)
    return predicted_labels


def calculate_metrics(test_labels, predicted_test_labels):
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, predicted_test_labels, average="weighted")
    accuracy = accuracy_score(test_labels, predicted_test_labels)
    logger.info("PRECISION: %.2f", precision)
    logger.info("RECALL: %.2f", recall)
    logger.info("F1 SCORE: %.2f", f1)
    logger.info("ACCURACY: %.2f", accuracy)
    return precision, recall, f1, accuracy


def train_model(df, model_name):
    logger.info("GETTING TEXTS AND LABELS")
    train_texts, train_labels = get_texts_and_labels(df)
    logger.info("SPLITTING")
    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.2)
    logger.info("TOKENIZING")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    logger.info("CREATING DATASETS")
    train_dataset = RelationsDataset(train_encodings, train_labels)
    val_dataset = RelationsDataset(val_encodings, val_labels)
    training_args = TrainingArguments(
        output_dir="../results",  # output directory
        num_train_epochs=3,  # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir="./logs",  # directory for storing logs
        logging_steps=10,
        save_total_limit=1,
        save_strategy="no",
        load_best_model_at_end=False,
    )
    labels_number = len(set(train_labels))
    model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=labels_number)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    logger.info("TRAINING")
    trainer.train()
    trainer.save_model(model_name)
    return model


def evaluate_model(test_df, model_name):
    test_texts, test_labels = get_texts_and_labels(test_df)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    labels_number = len(set(test_labels))
    logger.info("Loading model")
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=labels_number).to("cuda:0")
    generator = pipeline(task="text-classification", model=model, tokenizer=tokenizer, device=0)
    predicted_test_labels = generator(test_texts)
    predicted_test_labels = prune_prefixes_from_labels(predicted_test_labels)
    calculate_metrics(test_labels, predicted_test_labels)


def main():
    # TODO loading and parsing arguments, remove hardcoding
    train_filename = "../train.tsv"
    test_filename = "../test.tsv"
    model_name = "third-relations-tagged"
    subset_percentage = 0.5
    train_df = pd.read_csv(train_filename, sep="\t")
    test_df = pd.read_csv(test_filename, sep="\t")
    train_df = train_df.sample(frac=subset_percentage)
    train_model(train_df, model_name)
    evaluate_model(test_df, model_name)


if __name__ == "__main__":
    main()
