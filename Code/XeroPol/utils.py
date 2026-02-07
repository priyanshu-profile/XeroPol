import os
import random
import logging
from enum import Enum
import torch
import numpy as np
from transformers import XLMRobertaTokenizer, XLMRobertaModel, AutoTokenizer, AutoModelForMaskedLM
from xlm_r_ca import XeroPol, get_polite_labels, get_sentiment_labels
from sklearn.metrics import classification_report as sk_report, confusion_matrix, precision_score as p, recall_score as r, f1_score as f1

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'large': (XLMRobertaModel, XeroPol),
    'base': (XLMRobertaModel, XeroPol)
}

MODEL_PATH_MAP = {
    'large': 'xlm-roberta-large',
    'base': 'xlm-roberta-base'
}

class Tasks(Enum):
    EMOWOZ = 'emowoz'
    MULTIDOGO = 'multidogo'


def load_tokenizer(model_name_or_path):
    return XLMRobertaTokenizer.from_pretrained(model_name_or_path)
    
def init_logger(args):
    if not os.path.exists(os.path.join(args.model_dir)):
        os.mkdir(os.path.join(args.model_dir))
    # noinspection PyArgumentList
    logging.basicConfig(handlers=[logging.FileHandler(os.path.join(args.model_dir, "log.log"), "a+", "utf-8"), logging.StreamHandler()],
                        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda_device != "cpu" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(polite_predictions, polite_labels, sentiment_predictions, sentiment_labels, examples, guids, args):
    assert len(polite_predictions) == len(polite_labels)
    assert len(sentiment_predictions) == len(sentiment_labels) 
    pol_class_acc = {}
    sent_class_acc = {}
    polite_labels_list = get_polite_labels(args)
    sentiment_labels_list = get_sentiment_labels(args)
    polite_label_map = {i: label for i, label in enumerate(polite_labels_list)}
    sentiment_label_map = {i: label for i, label in enumerate(sentiment_labels_list)}
    polite_label = list(polite_label_map.keys())
    sentiment_label = list(sentiment_label_map.keys())
    pol_cm = confusion_matrix(polite_labels,polite_predictions,labels=polite_label)
    sent_cm = confusion_matrix(sentiment_labels,sentiment_predictions,labels=sentiment_label)

    pol_class_acc_list = pol_cm.diagonal()/pol_cm.sum(axis=1)
    sent_class_acc_list = sent_cm.diagonal()/sent_cm.sum(axis=1)
    
    for lbl,c_acc in zip(polite_label,pol_class_acc_list):
        pol_class_acc.update({polite_label_map[lbl]:c_acc})

    for lbl,c_acc in zip(sentiment_label,sent_class_acc_list):
        sent_class_acc.update({sentiment_label_map[lbl]:c_acc})
    
    results = {}
    results.update({"Politeness_Acc": (polite_predictions == polite_labels).mean()})
    results.update({"Sentiment_Acc": (sentiment_predictions == sentiment_labels).mean()})

    results.update({"Politeness_P (Macro)":p(polite_labels, polite_predictions, average='macro')})
    results.update({"Politeness_R (Macro)":r(polite_labels, polite_predictions, average='macro')})
    results.update({"Politeness_F1 (Macro)":f1(polite_labels, polite_predictions, average='macro')})
    results.update({"Politeness_P (Weighted)":p(polite_labels, polite_predictions, average='weighted')})
    results.update({"Politeness_R (Weighted)":r(polite_labels, polite_predictions, average='weighted')})
    results.update({"Politeness_F1 (Weighted)":f1(polite_labels, polite_predictions, average='weighted')})

    results.update({"Sentiment_P (Macro)":p(sentiment_labels, sentiment_predictions, average='macro')})
    results.update({"Sentiment_R (Macro)":r(sentiment_labels, sentiment_predictions, average='macro')})
    results.update({"Sentiment_F1 (Macro)":f1(sentiment_labels, sentiment_predictions, average='macro')})
    results.update({"Sentiment_P (Weighted)":p(sentiment_labels, sentiment_predictions, average='weighted')})
    results.update({"Sentiment_R (Weighted)":r(sentiment_labels, sentiment_predictions, average='weighted')})
    results.update({"Sentiment_F1 (Weighted)":f1(sentiment_labels, sentiment_predictions, average='weighted')})

    results.update({"Politeness Label Map": polite_label_map})
    results.update({"Sentiment Label Map": sentiment_label_map})

    results.update({"Politeness Confusion Matrix": pol_cm})
    results.update({"Politeness Class Accuracy": pol_class_acc})

    results.update({"Sentiment Confusion Matrix": sent_cm})
    results.update({"Sentiment Class Accuracy": sent_class_acc})
    return results
