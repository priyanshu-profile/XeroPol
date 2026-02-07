import copy
import os
import logging
import pickle
import torch
from torch.utils.data import TensorDataset
from xlm_r_ca import get_polite_labels, get_sentiment_labels 
from utils import Tasks

logger = logging.getLogger(__name__)


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


class XNLUProcessor(object):
    """ XNLUProcessor for all XNLU data sets """

    def __init__(self, args):
        self.args = args
        self.polite_labels = get_polite_labels(args)
        self.sentiment_labels = get_sentiment_labels(args)

    def _create_examples(self, data, start_id):
        examples = []
        inputs = data['inputs']
        polite_labels = data['polite_labels']
        sentiment_labels = data['sentiment_labels']
        for inp, pol_lbl, sen_lbl in zip(inputs, polite_labels, sentiment_labels):
            guid = start_id
            start_id += 1
            polite_label = self.polite_labels.index(pol_lbl)
            sentiment_label = self.sentiment_labels.index(sen_lbl)
        
            examples.append(InputExample(guid=guid, words=inp, polite_label=polite_label, sentiment_label=sentiment_label))
        return examples

    def get_examples(self, mode, languages):
        """
        Args:
            mode: train, dev, test
            languages: which languages to select
        """
        examples = []
        for language in languages.split(","):
            data_path = os.path.join(self.args.data_dir, self.args.task, language, mode)
            logger.info("Reading file: {}".format(data_path))
            examples.extend(self._create_examples(pickle.load(open(os.path.join(data_path, "data.pkl"), "rb")),
                                                  start_id=len(examples)))
        return examples

def convert_examples_to_features(examples, args, tokenizer):
    # Settings based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token_id = tokenizer.pad_token_id
    max_seq_len = args.max_seq_len
    pad_token_label_id = args.ignore_index

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

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
        
        polite_label_id = int(example.polite_label)
        sentiment_label_id = int(example.sentiment_label)

        if ex_index < 1:
            logger.info("*** Example %d ***" % ex_index)
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("polite_label: %d" % example.polite_label)
            logger.info("sentiment_label: %d" % example.sentiment_label)

        features.append(InputFeatures(input_ids=input_ids, polite_label=polite_label_id, sentiment_label=sentiment_label_id, guid=example.guid))

    return features


def load_and_cache_examples(args, tokenizer, mode):
    if args.task in [Tasks.EMOWOZ.value, Tasks.MULTIDOGO.value]:    
        processor = XNLUProcessor(args)
    

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.model_dir, 'features_{}_{}.bin'.format(mode, args.task))
    cached_examples_file = os.path.join(args.model_dir, 'examples_{}_{}.bin'.format(mode, args.task))
    assert args.task in [t.value for t in Tasks]

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        examples = torch.load(cached_examples_file)
    else:
        # Load data features from dataset file
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples("train", args.train_languages)
        elif mode == "dev":
            examples = processor.get_examples("dev", args.dev_languages)
        elif mode == "test":
            examples = processor.get_examples("test", args.test_languages)
        else:
            raise Exception("For mode, only train, dev, test is available...")

        if args.task in [Tasks.EMOWOZ.value, Tasks.MULTIDOGO.value]:
            features = convert_examples_to_features(examples=examples, args=args, tokenizer=tokenizer)
        else:
            raise Exception("Sorry, the task '%s' is not recognised." % args.task)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)
        logger.info("Saving examples into cached file %s", cached_examples_file)
        torch.save(examples, cached_examples_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_polite_label_ids = torch.tensor([f.polite_label for f in features], dtype=torch.long)
    all_sentiment_label_ids = torch.tensor([f.sentiment_label for f in features], dtype=torch.long)
    all_guids = torch.tensor([f.guid for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_polite_label_ids, all_sentiment_label_ids, all_guids)
    return dataset, examples
