import sys
from glob import glob

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments, pipeline,
)
import pandas as pd
from sklearn.model_selection import train_test_split


def compute_metrics(pred,average='micro'):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=average)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def preprocess_and_save(df, filename):
    df = df.replace('<e1>', '')
    df = df.replace('</e1>', '')
    df = df.replace('<e2>', '')
    df = df.replace('</e2>', '')
    df.to_csv(f"preprocessed_{filename}", sep='\t')


def get_texts_and_labels(filename):
    df = pd.read_csv(filename, sep="\t")
    # if filename == "train.tsv":
    #     df = df.sample(frac=1)
    #     df = df.sample(frac=0.20)
    # preprocess_and_save(df, filename)
    texts = df["text"].tolist()
    labels = df["label"].tolist()
    d = dict([(y, x) for x, y in enumerate(sorted(set(labels)))])
    labels = [d[x] for x in labels]
    return texts, labels


def convert_predictions(predictions):
    predicted_labels = list()
    for prediction in predictions:
        label = prediction['label']
        label = label.replace('LABEL_', "")
        label = int(label)
        predicted_labels.append(label)
    return predicted_labels


def evaluate_model(test_labels, predicted_test_labels):
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, predicted_test_labels, average="weighted")
    accuracy = accuracy_score(test_labels, predicted_test_labels)
    print("PRECISION:", precision)
    print("RECALL:", recall)
    print("F1 SCORE", f1)
    print("ACCURACY", accuracy)


def main():
    # print("GETTING TEXTS AND LABELS")
    # train_texts, train_labels = get_texts_and_labels("20_with_tags_train.tsv")
    #
    # print("SPLITTING")
    # train_texts, val_texts, train_labels, val_labels = train_test_split(
    #     train_texts, train_labels, test_size=0.2
    # )
    #
    # print("TOKENIZING")
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    # train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    # val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    #
    # print("CREATING DATASETS")
    # train_dataset = Dataset(train_encodings, train_labels)
    # val_dataset = Dataset(val_encodings, val_labels)
    #
    # training_args = TrainingArguments(
    #     output_dir="./results",  # output directory
    #     num_train_epochs=3,  # total number of training epochs
    #     per_device_train_batch_size=16,  # batch size per device during training
    #     per_device_eval_batch_size=64,  # batch size for evaluation
    #     warmup_steps=500,  # number of warmup steps for learning rate scheduler
    #     weight_decay=0.01,  # strength of weight decay
    #     logging_dir="./logs",  # directory for storing logs
    #     logging_steps=10,
    #     save_total_limit=2,
    #     save_strategy="no",
    #     load_best_model_at_end=False,
    # )
    # labels_number = len(set(train_labels))
    # model=BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased",num_labels=labels_number)
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=val_dataset,
    #     compute_metrics=compute_metrics
    # )
    # print("TRAINING")
    # trainer.train()
    # trainer.save_model('second-relations-tagged')

    # AFTER TRAINING
    model_name = 'second-relations'
    test_texts, test_labels = get_texts_and_labels("20_no_tags_test.tsv")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    labels_number = len(set(test_labels))
    print("Loading model")
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=labels_number).to('cuda:0')
    print("Running pipeline")
    generator = pipeline(task="text-classification", model=model, tokenizer=tokenizer, device=0)
    print("1")
    predicted_test_labels = generator(test_texts)
    print("2")
    predicted_test_labels = convert_predictions(predicted_test_labels)
    print("3")
    evaluate_model(test_labels, predicted_test_labels)


if __name__ == '__main__':
    main()
