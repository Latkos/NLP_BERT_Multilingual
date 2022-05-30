from seqeval.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

from logger import logger


def get_texts_and_labels(df):
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
