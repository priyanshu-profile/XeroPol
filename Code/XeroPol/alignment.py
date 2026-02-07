import logging
import os
import pickle
import torch
from torch.utils.data import TensorDataset
from data_loader import InputFeatures
from utils import load_tokenizer, Tasks

logger = logging.getLogger(__name__)


# noinspection PyCallingNonCallable
def generate_alignment_pairs(args):

    # Experiments could be faster by caching the alignment data
    # cached_features_file = os.path.join(args.model_dir, 'alignment_{}_{}.bin'.format(mode, args.task))

    examples = []

    for language in args.align_languages.split(","):
        with open(os.path.join(args.data_dir, args.task, language, "train.txt"), "r", encoding="utf-8") as tar_f, \
             open(os.path.join(args.data_dir, args.task, "en", "train.txt"), "r", encoding="utf-8") as eng_f:
            data = pickle.load(open(os.path.join(args.data_dir, args.task, "en", "train", "data.pkl"), "rb"))
            for target_line, english_line, polite_label, sentiment_label in zip(tar_f, eng_f, data['polite_labels'],data['sentiment_labels']):
                target_line = target_line.split("|")[3]
                english_line = english_line.split("|")[3]
                examples.append(((target_line.strip(), english_line.strip()), polite_label.strip(), sentiment_label.strip()))
        logger.info("Read %d lines...." % len(examples))

    examples = examples
    tokenizer = load_tokenizer(args.model_name_or_path)
    pad_token_id = tokenizer.pad_token_id

    feats = []
    for ex_id, example in enumerate(examples):
        if ex_id % 5000 == 0:
            logger.info("Processed %d examples..." % ex_id)
        input_ids = []

        if args.task in [Tasks.EMOWOZ.value, Tasks.MULTIDOGO.value]:
        
            for utterance in example[0]:
                
                tokens = tokenizer.tokenize(utterance)
                ids = tokenizer.build_inputs_with_special_tokens(tokenizer.convert_tokens_to_ids(tokens))
                if tokens is not None and len(ids) > args.max_seq_len:
                    print('Lengths',len(ids),len(tokens),tokens)
                assert tokens is not None and len(ids) <= args.max_seq_len
                ids = ids + ([pad_token_id] * (args.max_seq_len - len(ids)))
                input_ids.append(ids)
      
        else:
            raise Exception("The task '%s' is not recognised!" % args.task)

        if len(input_ids) == 2:
            feats.append(InputFeatures(input_ids, None, None, None))

    # Convert to Tensors and build dataset
    target_input_ids = torch.tensor([f.input_ids[0] for f in feats], dtype=torch.long)
    english_input_ids = torch.tensor([f.input_ids[1] for f in feats], dtype=torch.long)
    train_dataset = TensorDataset(target_input_ids, english_input_ids)
    assert len(target_input_ids) == len(english_input_ids)

    logger.info("Created %d train/align instances." % len(train_dataset))
    return train_dataset
