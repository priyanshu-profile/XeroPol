import os
import pickle
import torch.nn as nn
from transformers import XLMRobertaModel
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from loss_func import CELoss, SupConLoss

class SequenceClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SequenceClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


class XeroPol(BertPreTrainedModel):
    def __init__(self, config, args):
        super(XeroPol, self).__init__(config)
        self.args = args
        self.num_polite_labels = len(get_polite_labels(args))
        self.num_sentiment_labels = len(get_sentiment_labels(args))
        self.roberta = XLMRobertaModel(config=config)
        self.politeness_classifier = SequenceClassifier(config.hidden_size, self.num_polite_labels)
        self.sentiment_classifier = SequenceClassifier(config.hidden_size, self.num_sentiment_labels)
        
    def forward(self, input_ids, polite_labels=None,sentiment_labels=None):
        outputs = self.roberta(input_ids)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # <s>  #outputs[cls_feats]

        polite_logits = self.politeness_classifier(pooled_output)
        sentiment_logits = self.sentiment_classifier(pooled_output) #outputs['predicts]
        total_loss = 0

        if polite_labels is not None:
            if self.num_polite_labels == 1:
                polite_loss_fct = nn.MSELoss()
                polite_loss = polite_loss_fct(polite_logits.view(-1), polite_labels.view(-1))
            else:
                polite_loss_fct = nn.CrossEntropyLoss()
                polite_loss = polite_loss_fct(polite_logits.view(-1, self.num_polite_labels), polite_labels.view(-1))
            total_loss += polite_loss

        if sentiment_labels is not None:
            if self.num_sentiment_labels == 1:
                sentiment_loss_fct = nn.MSELoss()
                sentiment_loss = sentiment_loss_fct(sentiment_logits.view(-1), sentiment_labels.view(-1))
            else:
                if self.args.aux_loss == 'scl':
                    sentiment_loss_fct = SupConLoss(self.args.alpha, self.args.temp)
                    sentiment_loss = sentiment_loss_fct(sentiment_logits.view(-1, self.num_sentiment_labels), sentiment_labels.view(-1),cls_feats=pooled_output,predicts=sentiment_logits)
                else:
                    sentiment_loss_fct = nn.CrossEntropyLoss()
                    sentiment_loss = sentiment_loss_fct(sentiment_logits.view(-1, self.num_sentiment_labels), sentiment_labels.view(-1))
            total_loss += sentiment_loss

        outputs = ((polite_logits,sentiment_logits),) + outputs[2:]
        return (total_loss,) + outputs


def get_polite_labels(args):
    return pickle.load(open(os.path.join(args.data_dir, args.task, "polite_labels.pkl"), 'rb'))

def get_sentiment_labels(args):
    return pickle.load(open(os.path.join(args.data_dir, args.task, "sentiment_labels.pkl"), 'rb'))
