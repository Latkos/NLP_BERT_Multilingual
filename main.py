import torch
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

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

def main():
    en_small_train = pd.read_csv("data/en-small_corpora_train.tsv", sep="\t")
    en_small_test = pd.read_csv("data/en-small_corpora_test.tsv", sep="\t")

    print(en_small_train.columns.values)

    train_texts = en_small_train["text"].tolist()
    test_texts = en_small_test["text"].tolist()
    train_labels = en_small_train["label"].tolist()
    test_labels = en_small_test["label"].tolist()

    d = dict([(y, x) for x, y in enumerate(sorted(set(train_labels)))])
    labels_number=len(d)
    train_labels = [d[x] for x in train_labels]
    test_labels = [d[x] for x in test_labels]

    print("SPLITTING")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.2
    )

    print("TOKENIZING")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)


    print("CONVERTING TO DATASETS")
    train_dataset = Dataset(train_encodings, train_labels)
    val_dataset = Dataset(val_encodings, val_labels)
    test_dataset = Dataset(test_encodings, test_labels)

    print("PREPARING TRAINER")
    training_args = TrainingArguments(
        output_dir="./results",  # output directory
        num_train_epochs=3,  # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir="./logs",  # directory for storing logs
        logging_steps=10,
    )

    model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased",num_labels=labels_number)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    print("TRAINING")
    trainer.train()

if __name__=='__main__':
    main()
