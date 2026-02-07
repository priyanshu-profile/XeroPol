import os
import copy

import logging
import argparse
from tqdm import tqdm, trange

import numpy as np
import torch
from train import Trainer
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from xlm_r_ca import get_polite_labels,get_sentiment_labels
from utils import init_logger, load_tokenizer, get_polite_labels, MODEL_CLASSES, MODEL_PATH_MAP, Tasks, set_seed

logger = logging.getLogger(__name__)

def read_input_file(args):
    lines = []
    dialog_ids = []
    utt_ids = []
    speaker = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            temp = line.split('\t')
            if len(temp) == 4:
                line = temp[3].strip()
                words = line.split()
                lines.append(words)
                dialog_ids.append(temp[0])
                utt_ids.append(temp[1])
                speaker.append(temp[2])
    return lines, dialog_ids, utt_ids, speaker

class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        polite_label: (Optional) string. The politeness label of the example.
        sentiment_label: (Optional) list. The sentiment label of the example.
    """
    
    def __init__(self, guid, words, polite_label=None, sentiment_label=None):
        self.guid = guid
        self.words = words
        self.polite_label = polite_label
        self.sentiment_label = sentiment_label
        

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, polite_label, sentiment_label, guid):
        self.input_ids = input_ids
        self.polite_label = polite_label
        self.sentiment_label = sentiment_label
        self.guid = guid

def convert_examples_to_features(examples, args, tokenizer):
    # Settings based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token_id = tokenizer.pad_token_id
    max_seq_len = args.max_seq_len
    pad_token_label_id = args.ignore_index

    features = []
    for (ex_index, example) in enumerate(examples):

        tokens = copy.deepcopy(example.words)
        
        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            raise Exception("Increase max_seq_len, please!")

        # Add [SEP] token
        tokens += [sep_token]
        
        # Add [CLS] token
        tokens = [cls_token] + tokens
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        
        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)

        features.append(InputFeatures(input_ids=input_ids, polite_label=None, sentiment_label=None, guid=example.guid))

    return features


def convert_input_file_to_tensor_dataset(lines,
                                         pred_config,
                                         args,
                                         tokenizer):

    
    all_inputs = []
    

    for tokens in lines:
        inputs = []
        for token in tokens:
            tokenized = [t for t in tokenizer.tokenize(token) if t != '▁']
            for idx, t in enumerate(tokenized):
                inputs.append(t)
        all_inputs.append(inputs)
    
    examples = []
    start_id = len(examples)
    for inp in all_inputs:
        examples.append(InputExample(guid=start_id, words=inp, polite_label=None, sentiment_label=None))
        start_id+=1

    features = convert_examples_to_features(examples=examples, args=args, tokenizer=tokenizer)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_guids = torch.tensor([f.guid for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_guids)
    return dataset, examples

def predict(args):

    lines, dialog_ids, utt_ids, speaker = read_input_file(args)
    device = args.cuda_device if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer(args.model_name_or_path)
    load_model_path = args.model_dir
    encoder_class, model_class = MODEL_CLASSES[args.model_type]
    model_args = torch.load(os.path.join(args.model_dir, 'training_args.bin'))
    try:
        logger.info("*****************************************************")
        logger.info("***** Loading Model from '%s' *****" % load_model_path)
        model = model_class.from_pretrained(load_model_path, args=model_args)
        model.to(device)
        logger.info("*****************************************************")
    except Exception as e:
        raise Exception(e)

    polite_labels_lst = get_polite_labels(model_args)
    sentiment_labels_lst = get_sentiment_labels(model_args)
    
    dataset, examples = convert_input_file_to_tensor_dataset(lines, args, model_args, tokenizer)
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.batch_size)

    # Eval!
    logger.info("***** Running predictions*****")
    logger.info("  Num examples = %d" % len(dataset))
    logger.info("  Batch size = %d" % args.batch_size)
    model.eval()
    polite_preds_final = []
    sentiment_preds_final = []
    for batch in tqdm(eval_dataloader, desc="Predicting...", disable=True):
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0]}
                outputs = model(**inputs)
                _, (polite_logits,sentiment_logits) = outputs[:2]

            polite_preds = polite_logits.detach().cpu().numpy()
            sentiment_preds = sentiment_logits.detach().cpu().numpy()
            polite_preds = np.argmax(polite_preds, axis=1)
            polite_preds_final.extend(polite_preds)
            sentiment_preds = np.argmax(sentiment_preds, axis=1)
            sentiment_preds_final.extend(sentiment_preds)
    
    # Write to output file
    with open(args.output_file, "w", encoding="utf-8") as f:
        for conv, utt, s, words, polite_pred, sentiment_pred in zip(dialog_ids, utt_ids, speaker, lines, polite_preds_final,sentiment_preds_final):
            utternace = ' '.join(words)
            p_label = polite_labels_lst[polite_pred]
            s_label = sentiment_labels_lst[sentiment_pred]
            f.write(conv+'\t'+utt+'\t'+s+'\t'+utternace+'\t'+p_label+'\t'+s_label+'\n')
    logger.info("Prediction Done!")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", default="predictions/sample_pred_in.txt", type=str, help="Input file for prediction")
    parser.add_argument("--output_file", default="predictions/sample_pred_out.txt", type=str, help="Output file for prediction")
    parser.add_argument("--model_type", required=True, type=str,
                        help="Model type selected from the following list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_dir", default="./multidogo_aligned_hi", type=str, help="Path to load model")
    parser.add_argument('--seed', type=int, default=123456789,help="Random seed for model initialization")

    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for prediction")
    parser.add_argument("--cuda_device", default="cpu", type=str, help="Which CUDA device to use or 'cpu' for CPU")    
    args = parser.parse_args()
    init_logger(args)
    set_seed(args)
    args.model_name_or_path = MODEL_PATH_MAP[args.model_type]
    predict(args)