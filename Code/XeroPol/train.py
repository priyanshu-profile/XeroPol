import os
import random
import logging
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.adam import Adam
from torch.nn import MSELoss, CrossEntropyLoss, CosineEmbeddingLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from xlm_r_ca import get_polite_labels, get_sentiment_labels
from utils import MODEL_CLASSES, compute_metrics, Tasks

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, args, train_dataset, dev_dataset, test_dataset,
                 train_examples, dev_examples, test_examples, tokenizer, alignment_dataset):
        self.args = args
        self.alignment_dataset = alignment_dataset
        self.train_dataset = train_dataset
        self.train_examples = train_examples
        self.dev_dataset = dev_dataset
        self.dev_examples = dev_examples
        self.test_dataset = test_dataset
        self.test_examples = test_examples
        self.tokenizer = tokenizer
        self.pad_token_id = args.ignore_index
        self.encoder_class, self.model_class = MODEL_CLASSES[args.model_type]
        self.device = args.cuda_device if torch.cuda.is_available() else "cpu"
        self.model = None  # assigned in the load_model() function

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)

        t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs
        
        optimizer = Adam(self.model.parameters(), lr=self.args.learning_rate)
        scheduler = OneCycleLR(optimizer, max_lr=self.args.learning_rate, total_steps=t_total)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Train batch size = %d", self.args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        tr_loss, align_loss = 0.0, 0.0
        best_dev_loss = float('inf')
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")

        for e in train_iterator:
            logging.info("****************** Epoch %d ********************" % (e))
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU

                if self.args.task in [Tasks.EMOWOZ.value, Tasks.MULTIDOGO.value]:
                    inputs = {'input_ids': batch[0], 'polite_labels': batch[1],'sentiment_labels': batch[2]}
                else:
                    raise Exception("The task name '%s' is not recognised/supported." % self.args.task)

                outputs = self.model(**inputs)
                loss = outputs[0]
                tr_loss += loss.item()

                if self.args.align_languages:

                    if self.args.align_loss == 'ncal':

                        encoder = self.model.roberta
                        indices = random.sample(range(len(self.alignment_dataset)), self.args.train_batch_size)

                        batch_one = torch.stack([self.alignment_dataset[index][0] for index in indices]).to(self.device)
                        outputs = encoder(input_ids=batch_one)
                        cls_logits = outputs[1]

                        batch_two = torch.stack([self.alignment_dataset[index][1] for index in indices]).to(self.device)
                        outputs = encoder(input_ids=batch_two)
                        cls_target = outputs[1]

                        #loss_fn = MSELoss()
                        #xero_align_loss = loss_fn(input=cls_logits, target=cls_target)
                        
                        loss_fn = CosineEmbeddingLoss()
                        y = torch.ones(cls_logits.shape[0], device=self.args.cuda_device)
                        xero_align_loss = loss_fn(cls_logits, cls_target, y)
                        
                        align_loss += xero_align_loss.item()
                        loss += xero_align_loss
                    else:
                        # ---- Contrastive Loss ----
                        encoder = self.model.roberta
                        indices = random.sample(range(len(self.alignment_dataset)), self.args.train_batch_size)

                        batch_one = torch.stack([self.alignment_dataset[index][0] for index in indices]).to(self.device)
                        outputs = encoder(input_ids=batch_one)
                        cls_logits = outputs[1]

                        batch_two = torch.stack([self.alignment_dataset[index][1] for index in indices]).to(self.device)
                        outputs = encoder(input_ids=batch_two)
                        cls_target = outputs[1]

                        # Contrastive (CLS) representation
                        src_outputs = cls_logits
                        tar_outputs = cls_target

                        src_n, tar_n = src_outputs.norm(dim=1)[:, None], tar_outputs.norm(dim=1)[:, None]
                        a_norm = src_outputs / src_n
                        b_norm = tar_outputs / tar_n
                        sim = torch.mm(a_norm, b_norm.transpose(0, 1)).to(self.device)

                        labels = torch.arange(sim.shape[0]).to(self.device)

                        loss_fn = CrossEntropyLoss()
                        contrastive_loss = loss_fn(input=sim, target=labels)
                        align_loss += contrastive_loss.item()
                        loss += contrastive_loss
                        
                        # ---- Contrastive Loss ----

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()  # Only do this once for all losses

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()
                    global_step += 1

            if self.args.task in [Tasks.EMOWOZ.value, Tasks.MULTIDOGO.value]:
                self.evaluate_xnlu("dev", exp_name=self.args.model_dir)
            else:
                raise Exception("The task name '%s' is not recognised/supported." % self.args.task)
            logging.info("--------------------------------------")
            logging.info("Train loss after %d steps: %.3f" % (global_step, (tr_loss / global_step)))
            if self.args.align_languages:
                logging.info("Align Loss after %d steps: %.3f" % (global_step, align_loss / max(1, global_step)))
            logging.info("--------------------------------------")



    def evaluate_xnlu(self, mode, exp_name):
        if mode == 'test':
            dataset = self.test_dataset
            examples = self.test_examples
        elif mode == 'dev':
            dataset = self.dev_dataset
            examples = self.dev_examples
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****" % mode)
        logger.info("  Num examples = %d" % len(dataset))
        logger.info("  Batch size = %d" % self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        polite_preds = None
        sentiment_preds = None
        guids = None
        out_polite_labels = None
        out_sentiment_labels = None
        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating...", disable=True):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0], 'polite_labels': batch[1], 'sentiment_labels': batch[2]}
                outputs = self.model(**inputs)
                tmp_eval_loss, (polite_logits,sentiment_logits) = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            # Collect ids for debugging
            if guids is None:
                guids = batch[3].detach().cpu().numpy()
            else:
                guids = np.append(guids, batch[3].detach().cpu().numpy(), axis=0)

            # Politeness prediction
            if polite_preds is None:
                polite_preds = polite_logits.detach().cpu().numpy()
                out_polite_labels = inputs['polite_labels'].detach().cpu().numpy()
            else:
                polite_preds = np.append(polite_preds, polite_logits.detach().cpu().numpy(), axis=0)
                out_polite_labels = np.append(out_polite_labels, inputs['polite_labels'].detach().cpu().numpy(), axis=0)

            # Sentiment prediction
            if sentiment_preds is None:
                sentiment_preds = sentiment_logits.detach().cpu().numpy()
                out_sentiment_labels = inputs['sentiment_labels'].detach().cpu().numpy()
            else:
                sentiment_preds = np.append(sentiment_preds, sentiment_logits.detach().cpu().numpy(), axis=0)
                out_sentiment_labels = np.append(out_sentiment_labels, inputs['sentiment_labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {"loss": eval_loss}

        # Politeness result
        polite_preds = np.argmax(polite_preds, axis=1)

        # Sentiment result
        sentiment_preds = np.argmax(sentiment_preds, axis=1)

        total_result = compute_metrics(polite_preds, out_polite_labels, sentiment_preds, out_sentiment_labels, examples, guids, self.args)
        results.update(total_result)

        logger.info("********* %s results for %s *********" % (mode.capitalize(), exp_name))
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
        logger.info("********* %s results for %s *********" % (mode.capitalize(), exp_name))
        # return results


    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        self.model.save_pretrained(self.args.model_dir)

        # Save training arguments together with the trained model
        torch.save(self.args, os.path.join(self.args.model_dir, 'training_args.bin'))
        logger.info("Saved model checkpoint to '%s'" % self.args.model_dir)

    def load_model(self, final_eval):
        # Okay, there a few options for loading models...
        if final_eval:
            load_model_path = self.args.load_eval_model if self.args.load_eval_model else self.args.model_dir
        elif self.args.load_train_model:
            load_model_path = self.args.load_train_model
        else:
            load_model_path = self.args.model_name_or_path
        """
        if not os.path.exists(load_model_path):
            raise Exception("Model path '%s' doesn't exists! Train first, please..." % load_model_path)
        """
        try:
            logger.info("*****************************************************")
            logger.info("***** Loading Model from '%s' *****" % load_model_path)
            self.model = self.model_class.from_pretrained(load_model_path, args=self.args)
            self.model.to(self.device)
            logger.info("*****************************************************")
        except Exception as e:
            raise Exception(e)
