from report.pretrain.Pooler import *
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import logging

def get_pooler_class(pooler_type, **kwargs):
    if pooler_type == "cls":
        return ClsPooling(**kwargs)
    elif pooler_type == "mean":
        return MeanPooling(**kwargs)
    elif pooler_type == "max":
        return MaxPooling(**kwargs)

def get_metrics(metrics_type, **kwargs):
    if metrics_type == "precision_score":
        return precision_score
    elif metrics_type == "recall_score":
        return recall_score
    elif metrics_type == "f1_score":
        return f1_score
    elif metrics_type == "accuracy_score":
        return accuracy_score

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("debug.log"),
            logging.StreamHandler()
        ]
    )