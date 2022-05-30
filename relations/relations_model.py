from sklearn.model_selection import train_test_split
from transformers import Trainer, BertForSequenceClassification, AutoTokenizer, TrainingArguments, pipeline

from helper_functions_relations import get_texts_and_labels, prune_prefixes_from_labels, calculate_metrics
from logger import logger
from relations_dataset import RelationsDataset


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
        output_dir="results",  # output directory
        num_train_epochs=3,  # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
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