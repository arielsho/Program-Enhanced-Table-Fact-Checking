from __future__ import absolute_import, division, print_function
import os
from collections import OrderedDict
import numpy as np
import argparse
import logging
import random
import sys
import csv
import pdb
import io
import json
import torch
import time
import shutil
import pickle as pkl
from pprint import pprint
from tqdm import tqdm, trange
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from tensorboardX import SummaryWriter
from models_BERT import BertForSequenceClassification, BertConfig
from models import DoubleBERT
logger = logging.getLogger(__name__)


class InputExample(object):
    def __init__(self, guid, table, fact=None, label=None):
        self.guid = guid
        self.table = table
        self.fact = fact
        self.label = label


class InputExample_verb(object):
    def __init__(self, guid, verb_num, verb_lst=None):
        self.guid = guid
        self.verb_num = verb_num
        self.verb_lst = verb_lst


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id, guid):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.guid = guid


class InputFeatures_verb(object):
    def __init__(self, input_ids, input_mask, segment_ids, guid, cls_ids, cls_mask, entity_ids=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.cls_ids = cls_ids
        self.cls_mask = cls_mask
        self.guid = guid
        self.entity_ids = entity_ids


class DataProcessor(object):
    def get_examples(self, data_dir, data_dir_verb, dataset, dataset_verb):
        lines, lines_verb = self._read_tsv(
            os.path.join(data_dir, dataset + '.tsv'),
            os.path.join(data_dir_verb, dataset_verb + '.tsv'),
        )
        return self._create_examples(lines, lines_verb)

    def get_labels(self):
        return ["0", "1"]

    def _get_table_verb(self, lst):
        lst_tmp = []
        if len(lst) == 0 or lst[0] == '0':
            lst_tmp.append("no program")
        else:
            for item in lst[1:]:
                lst_tmp.append(item + ' .')
        return lst_tmp

    def _read_tsv(self, input_file, input_file_verb, quotechar=None):
        lines = []
        lines_verb = []
        with open(input_file, "r", encoding="utf-8") as f:
            logger.info(">>> Reading: %s" % input_file)
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
        with open(input_file_verb, "r", encoding="utf-8") as f:
            logger.info(">>> Reading: %s" % input_file_verb)
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            for idx, line in enumerate(reader):
                assert line[0] == lines[idx][0]
                line_tmp = self._get_table_verb(line[4:-2])
                lines_verb.append(line_tmp)
        return lines, lines_verb

    def _create_examples(self, lines, lines_verb):
        examples_table = []
        examples_verb = []

        for (i, line) in enumerate(lines):
            if args.debug and i > 1000:
                break
            guid = line[0]
            try:
                column_types = line[2].split()
                table = line[3]
                fact = line[4]
                label = line[5]
                verb = lines_verb[i]
            except IndexError:
                continue
            examples_table.append(InputExample(guid=guid, fact=fact, table=table, label=label))
            examples_verb.append(InputExample_verb(guid=guid, verb_num=len(verb), verb_lst=verb))

        return examples_table, examples_verb


def convert_examples_to_features(examples_table, examples_verb, label_list, max_seq_length,
                                 tokenizer, fact_place=None, balance=False, verbose=False, phase=None):
    assert fact_place is not None
    label_map = {"0": 0, "1": 1}

    features_table = []
    features_verb = []
    logger.info("convert_examples_to_features ...\n")
    max_table = 0
    avg_table = 0
    max_verb = 0
    avg_verb = 0
    for (ex_index, example) in enumerate(tqdm(examples_table)):
        guid = example.guid
        tokens_a = tokenizer.tokenize(example.table)  # table
        tokens_b = tokenizer.tokenize(example.fact)   # fact
        _truncate_seq_pair(tokens_a, tokens_b, args.max_seq_length_table - 3)

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * (len(tokens_a) + 2)
        assert len(tokens) == len(segment_ids)
        tokens += tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)

        if len(tokens) > max_table:
            max_table = len(tokens)
        avg_table += len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        padding = [0] * (args.max_seq_length_table - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == args.max_seq_length_table
        assert len(input_mask) == args.max_seq_length_table
        assert len(segment_ids) == args.max_seq_length_table

        label_id = label_map[example.label]

        # if ex_index < 1:
        #     logger.info("\n*** Example Table ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("label: %s (id = %d)" % (example.label, label_id))

        features_table.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          guid=guid))

    for (ex_index, example) in enumerate(tqdm(examples_verb)):
        guid = example.guid
        verb_num = example.verb_num
        if verb_num <= args.max_verb:
            pad_verb(example.verb_lst)
        else:
            pdb.set_trace()
            example.verb_lst = example.verb_lst[:args.max_verb]
            verb_num = args.max_verb

        tokens_tmp = []
        tokens_tmp_star = []

        for idx, v in enumerate(example.verb_lst):
            v_tmp_star = tokenizer.tokenize(v)
            tokens_tmp_star.append(v_tmp_star)

            v_tmp = tokenizer.tokenize(v.replace('*', ''))
            tokens_tmp.append(v_tmp)

        _truncate_seq_pair_verb(tokens_tmp, args.max_seq_length_verb - args.max_verb*2)
        _truncate_seq_pair_verb(tokens_tmp_star, args.max_seq_length_verb - args.max_verb * 2)

        tokens = []
        segment_ids = []
        cls_ids = []
        cls_mask = []
        for i in range(args.max_verb):
            cls_mask.append(0)
        for i in range(verb_num):
            cls_mask[i] = 1

        entity_ids = []
        for idx, item in enumerate(tokens_tmp):

            cls_ids.append(len(tokens))
            e_start_id = len(tokens)
            tokens.extend(["[CLS]"] + item + ["[SEP]"])
            segment_ids += [idx % 2] * (len(item) + 2)
            assert len(tokens) == len(segment_ids)

            item_star = tokens_tmp_star[idx]
            e_id = [1, 1]
            count = 0
            for n, word in enumerate(item_star):
                if word == '*':
                    count += 1
                    e_id[count-1] = n + 2 - count
                    if count == 2:
                        break
            if count == 1:
                e_id[1] = e_id[0]
            entity_ids.append(e_start_id + e_id[0])
            entity_ids.append(e_start_id + e_id[1])

        if len(tokens) > max_verb:
            max_verb = len(tokens)
        avg_verb += len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        padding = [0] * (args.max_seq_length_verb - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == args.max_seq_length_verb
        assert len(input_mask) == args.max_seq_length_verb
        assert len(segment_ids) == args.max_seq_length_verb

        # if ex_index < 1:
        #     logger.info("\n---  VERB Example --- ")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

        features_verb.append(InputFeatures_verb(input_ids=input_ids,
                                                input_mask=input_mask,
                                                segment_ids=segment_ids,
                                                guid=guid,
                                                cls_ids=cls_ids,
                                                cls_mask=cls_mask,
                                                entity_ids=entity_ids))  # TODO

    logger.info(">> max_table: {}, avg_table: {}".format(max_table, avg_table/len(features_table)))
    logger.info(">> max_verb: {}, avg_verb: {}".format(max_verb, avg_verb / len(features_verb)))
    logger.info(">> max len of verb ids: {}".format(args.max_seq_length_verb))

    return features_table, features_verb


def pad_verb(verb_lst):
    while len(verb_lst) < args.max_verb:
        verb_lst.append("[PAD]")


def _truncate_seq_pair_verb(tokens_tmp, max_length):
    while True:
        total_length = 0
        max_idx = 0
        max_idx_next = 0
        max_len = 0
        for i, tokens in enumerate(tokens_tmp):
            total_length += len(tokens)
            if len(tokens) > max_len:
                max_idx_next = max_idx
                max_len = len(tokens)
                max_idx = i
        if total_length <= max_length:
            break
        if len(tokens_tmp[max_idx]) > 2:
            tokens_tmp[max_idx].pop()
        else:
            tokens_tmp[max_idx_next].pop()


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    acc = 0
    for x, y in zip(preds, labels):
        if x == y:
            acc += 1

    return acc*1.0/len(preds)


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        # "f1": f1,
        # "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return acc_and_f1(preds, labels)


def get_dataloader(features_table, features_verb, phase=None):

    attn_file_dict = {'train': 'train_margin_attn_matrix', 'dev': 'dev_margin_attn_matrix',
                      'test': 'test_margin_attn_matrix', 'simple': 'simple_test_margin_attn_matrix',
                      'complex': 'complex_test_margin_attn_matrix', 'small': 'small_test_margin_attn_matrix'}

    attn_file_verb_entity = attn_file_dict[phase] + '_verb_entity.pth'
    attn_file_verb_verb = attn_file_dict[phase] + '_verb_verb.pth'
    attn_file_cls_verb = attn_file_dict[phase] + '_cls_verb.pth'

    graph_attn_verb_entity = torch.load(os.path.join(args.data_dir_verb, attn_file_verb_entity))
    graph_attn_verb_verb = torch.load(os.path.join(args.data_dir_verb, attn_file_verb_verb))
    graph_attn_cls_verb = torch.load(os.path.join(args.data_dir_verb, attn_file_cls_verb))

    if args.debug:
        graph_attn_verb_entity = graph_attn_verb_entity[:1001]
        graph_attn_verb_verb = graph_attn_verb_verb[:1001]
        graph_attn_cls_verb = graph_attn_cls_verb[:1001]

    all_graph_attn_verb_entity = torch.stack(graph_attn_verb_entity, dim=0)
    all_graph_attn_verb_verb = torch.stack(graph_attn_verb_verb, dim=0)
    all_graph_attn_cls_verb = torch.stack(graph_attn_cls_verb, dim=0)

    all_input_ids_table = torch.tensor([f.input_ids for f in features_table], dtype=torch.long)
    all_input_mask_table = torch.tensor([f.input_mask for f in features_table], dtype=torch.long)
    all_segment_ids_table = torch.tensor([f.segment_ids for f in features_table], dtype=torch.long)
    all_input_ids_verb = torch.tensor([f.input_ids for f in features_verb], dtype=torch.long)
    all_input_mask_verb = torch.tensor([f.input_mask for f in features_verb], dtype=torch.long)
    all_segment_ids_verb = torch.tensor([f.segment_ids for f in features_verb], dtype=torch.long)
    all_cls_ids = torch.tensor([f.cls_ids for f in features_verb], dtype=torch.long)
    all_entity_ids = torch.tensor([f.entity_ids for f in features_verb], dtype=torch.long)
    all_cls_mask = torch.tensor([f.cls_mask for f in features_verb], dtype=torch.float)
    guid_list = [f.guid for f in features_table]

    idx2guid_dict = {}
    for idx, guid in enumerate(guid_list):
        idx2guid_dict[idx] = guid
    all_guid = torch.tensor([k for k in range(len(idx2guid_dict.keys()))], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features_table], dtype=torch.long)
    all_data = TensorDataset(all_input_ids_table, all_input_mask_table, all_segment_ids_table,
                             all_input_ids_verb, all_input_mask_verb, all_segment_ids_verb,
                             all_cls_ids, all_cls_mask, all_label_ids, all_guid,
                             all_graph_attn_verb_entity, all_graph_attn_verb_verb, all_graph_attn_cls_verb, all_entity_ids)

    sampler = RandomSampler(all_data) if "train" in phase else SequentialSampler(all_data)
    dataloader = DataLoader(all_data, sampler=sampler, batch_size=args.train_batch_size if phase == "train" else args.eval_batch_size)

    return dataloader


def evaluate(args, model, device, processor, label_list, num_labels, tokenizer, output_mode, tr_loss, global_step,
             tbwriter=None, save_dir=None, load_step=0, event_switch=True, save_every_period=False):

    model.eval()
    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):

        eval_examples_table, eval_examples_verb = processor.get_examples(args.data_dir, args.data_dir_verb,
                                                                         args.test_set, args.test_set_verb)

        eval_features_table, eval_features_verb = convert_examples_to_features(eval_examples_table, eval_examples_verb,
                                                                               label_list, args.max_seq_length, tokenizer, fact_place=args.fact, balance=False, phase="dev")
        logger.info("***** Running eval *****")
        logger.info("  Num examples = %d", len(eval_examples_table))

        eval_dataloader = get_dataloader(eval_features_table, eval_features_verb, phase="dev")

        all_label_ids = torch.tensor([f.label_id for f in eval_features_table], dtype=torch.long)

        batch_idx = 0
        eval_loss = 0
        nb_eval_steps = 0
        preds = []
        temp = []

        input_id_lst = []
        logit_lst = []
        label_lst = []

        for step, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
            batch = tuple(t.to(device) for t in batch)
            input_ids_t, input_mask_t, segment_ids_t, input_ids_v, input_mask_v, segment_ids_v, cls_ids, cls_mask, \
            label_ids, guid_ids, graph_attn_verb_entity, graph_attn_verb_verb, graph_attn_cls_verb, entity_ids = batch

            with torch.no_grad():
                logits = model(input_ids_v, segment_ids_v, input_mask_v, input_ids_t, segment_ids_t, input_mask_t,
                               cls_ids, cls_mask, graph_attn_verb_entity, graph_attn_verb_verb, graph_attn_cls_verb, entity_ids, labels=None)

            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

            logit_lst.extend(logits.view(-1, num_labels).tolist())
            label_lst.extend(label_ids.view(-1).tolist())
            input_id_lst.extend(guid_ids.view(-1).tolist())

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(
                    preds[0], logits.detach().cpu().numpy(), axis=0)

            labels = label_ids.detach().cpu().numpy().tolist()
            start = batch_idx*args.eval_batch_size
            end = start+len(labels)
            batch_range = list(range(start, end))
            csv_names = [eval_examples_table[i].guid for i in batch_range]
            facts = [eval_examples_table[i].fact for i in batch_range]
            labels = label_ids.detach().cpu().numpy().tolist()
            assert len(csv_names) == len(facts) == len(labels)

            temp.extend([(x, y, z) for x, y, z in zip(csv_names, facts, labels)])
            batch_idx += 1

        eval_loss = eval_loss / nb_eval_steps
        preds = preds[0]
        if output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif output_mode == "regression":
            preds = np.squeeze(preds)

        evaluation_results = OrderedDict()
        for x, y in zip(temp, preds):
            c, f, l = x
            if not c in evaluation_results:
                evaluation_results[c] = [{'fact': f, 'gold': int(l), 'pred': int(y)}]
            else:
                evaluation_results[c].append({'fact': f, 'gold': int(l), 'pred': int(y)})

        if save_every_period:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            output_eval_file = os.path.join(save_dir, "eval_{}_results.json".format(args.test_set))
            with io.open(output_eval_file, "w", encoding='utf-8') as fout:
                json.dump(evaluation_results, fout, sort_keys=True, indent=4)

        result = compute_metrics(preds, all_label_ids.numpy())
        loss = tr_loss/args.period if args.do_train and global_step > 0 else None

        log_step = global_step if args.do_train and global_step > 0 else load_step
        result['eval_loss'] = eval_loss
        result['global_step'] = log_step
        result['loss'] = loss

        logger.info(">>>> eval results: {} ".format(args.test_set))
        for key in sorted(result.keys()):
            if event_switch and result[key] is not None and tbwriter is not None:
                tbwriter.add_scalar('{}/{}'.format(args.test_set, key), result[key], log_step)
            logger.info("  %s = %s", key, str(result[key]))

        if save_every_period:
            output_eval_metrics = os.path.join(save_dir, "eval_metrics.txt")
            with open(output_eval_metrics, "a") as writer:
                for key in sorted(result.keys()):
                    writer.write("%s = %s\n" % (key, str(result[key])))
        model.train()

        return result['acc']


def test(args, model, device, processor, label_list, num_labels, tokenizer, output_mode, tr_loss, global_step,
             tbwriter=None, save_dir=None, load_step=0, save_detailed_test=False, save_every_period=False):
    model.eval()
    if args.do_test and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples_table, eval_examples_verb = processor.get_examples(args.data_dir, args.data_dir_verb, args.plus_test_set, args.plus_test_set_verb)
        eval_features_table, eval_features_verb = convert_examples_to_features(
            eval_examples_table, eval_examples_verb, label_list, args.max_seq_length, tokenizer, fact_place=args.fact, balance=False, phase="test")
        logger.info("***** Running standard test *****")
        logger.info("  Num examples = %d", len(eval_examples_table))

        eval_dataloader = get_dataloader(eval_features_table, eval_features_verb, phase="test")

        all_label_ids = torch.tensor([f.label_id for f in eval_features_table], dtype=torch.long)

        batch_idx = 0
        eval_loss = 0
        nb_eval_steps = 0
        preds = []
        temp = []

        input_id_lst = []
        logit_lst = []
        label_lst = []

        for step, batch in enumerate(tqdm(eval_dataloader, desc="Testing")):
            batch = tuple(t.to(device) for t in batch)
            input_ids_t, input_mask_t, segment_ids_t, input_ids_v, input_mask_v, segment_ids_v, cls_ids, cls_mask, \
            label_ids, guid_ids, graph_attn_verb_entity, graph_attn_verb_verb, graph_attn_cls_verb, entity_ids = batch

            with torch.no_grad():
                logits = model(input_ids_v, segment_ids_v, input_mask_v, input_ids_t, segment_ids_t, input_mask_t,
                               cls_ids, cls_mask, graph_attn_verb_entity, graph_attn_verb_verb, graph_attn_cls_verb, entity_ids, labels=None)

            if output_mode == "classification":
                loss_fct = CrossEntropyLoss()
                tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
                logit_lst.extend(logits.view(-1, num_labels).tolist())
                label_lst.extend(label_ids.view(-1).tolist())
                input_id_lst.extend(guid_ids.view(-1).tolist())

            elif output_mode == "regression":
                loss_fct = MSELoss()
                tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(
                    preds[0], logits.detach().cpu().numpy(), axis=0)

            labels = label_ids.detach().cpu().numpy().tolist()
            start = batch_idx*args.eval_batch_size
            end = start+len(labels)
            batch_range = list(range(start, end))
            csv_names = [eval_examples_table[i].guid.replace("{}-".format(args.plus_test_set), "") for i in batch_range]
            facts = [eval_examples_table[i].fact for i in batch_range]
            labels = label_ids.detach().cpu().numpy().tolist()
            assert len(csv_names) == len(facts) == len(labels)

            temp.extend([(x, y, z) for x, y, z in zip(csv_names, facts, labels)])
            batch_idx += 1

        eval_loss = eval_loss / nb_eval_steps
        preds = preds[0]
        if output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif output_mode == "regression":
            preds = np.squeeze(preds)

        evaluation_results = OrderedDict()
        for x, y in zip(temp, preds):
            c, f, l = x
            if not c in evaluation_results:
                evaluation_results[c] = [{'fact': f, 'gold': int(l), 'pred': int(y)}]
            else:
                evaluation_results[c].append({'fact': f, 'gold': int(l), 'pred': int(y)})

        if save_every_period:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            output_eval_file = os.path.join(save_dir, "standard-test_{}_results.json".format(args.plus_test_set))
            with io.open(output_eval_file, "w", encoding='utf-8') as fout:
                json.dump(evaluation_results, fout, sort_keys=True, indent=4)

        result = compute_metrics(preds, all_label_ids.numpy())
        loss = tr_loss/args.period if args.do_train and global_step > 0 else None

        log_step = global_step if args.do_train and global_step > 0 else load_step
        result['eval_loss'] = eval_loss
        result['global_step'] = log_step
        result['loss'] = loss

        if save_every_period:
            output_eval_metrics = os.path.join(save_dir, "standard-test_metrics.txt")
            with open(output_eval_metrics, "a") as writer:
                for key in sorted(result.keys()):
                    writer.write("%s = %s\n" % (key, str(result[key])))

        logger.info(">>>> standard test results: {}".format(args.plus_test_set))
        for key in sorted(result.keys()):
            if result[key] is not None and tbwriter is not None:
                tbwriter.add_scalar('{}/{}'.format(args.plus_test_set, key), result[key], log_step)
            logger.info("  %s = %s", key, str(result[key]))

        if save_detailed_test:
            assert len(logit_lst) == len(preds)
            eval_res = OrderedDict()
            for x, y, logits in zip(temp, preds, logit_lst):
                c, f, l = x
                if not c in eval_res:
                    eval_res[c] = [{'fact': f, 'gold': int(l), 'pred': int(y), 'pred_logits': logits}]
                else:
                    eval_res[c].append({'fact': f, 'gold': int(l), 'pred': int(y), 'pred_logits': logits})
            output_eval_file = os.path.join(args.output_dir, "detailed_standard_{}_results.json".format(args.plus_test_set))
            logger.info("  Get new best acc on eval! Save the detailed test results: {}".format(output_eval_file))
            with io.open(output_eval_file, "w", encoding='utf-8') as fout:
                json.dump(eval_res, fout, sort_keys=True, indent=4)
    model.train()

def other_test(type, args, model, device, processor, label_list, num_labels, tokenizer, output_mode, tr_loss, global_step):
    logger.info("\n********** Running {}_test ************".format(type))
    if type == "simple": test_dataset = "simple_test"
    elif type == "complex": test_dataset = "complex_test"
    elif type == "small": test_dataset = "small_test"
    if type == "simple": test_dataset_verb = "simple_test_margin"
    elif type == "complex": test_dataset_verb = "complex_test_margin"
    elif type == "small": test_dataset_verb = "small_test_margin"

    model.eval()

    if args.local_rank == -1 or torch.distributed.get_rank() == 0:
        eval_examples_table, eval_examples_verb = processor.get_examples(args.data_dir, args.data_dir_verb, test_dataset, test_dataset_verb)
        eval_features_table, eval_features_verb = convert_examples_to_features(
            eval_examples_table, eval_examples_verb, label_list, args.max_seq_length, tokenizer, fact_place=args.fact, balance=False, phase=type)
        logger.info("  Num examples = %d", len(eval_examples_table))

        eval_dataloader = get_dataloader(eval_features_table, eval_features_verb, phase=type)

        all_label_ids = torch.tensor([f.label_id for f in eval_features_table], dtype=torch.long)

        batch_idx = 0
        eval_loss = 0
        nb_eval_steps = 0
        preds = []
        temp = []

        input_id_lst = []
        logit_lst = []
        label_lst = []

        for step, batch in enumerate(tqdm(eval_dataloader, desc=type)):
            batch = tuple(t.to(device) for t in batch)
            input_ids_t, input_mask_t, segment_ids_t, input_ids_v, input_mask_v, segment_ids_v, cls_ids, cls_mask, \
            label_ids, guid_ids, graph_attn_verb_entity, graph_attn_verb_verb, graph_attn_cls_verb, entity_ids = batch

            with torch.no_grad():
                logits = model(input_ids_v, segment_ids_v, input_mask_v, input_ids_t, segment_ids_t, input_mask_t,
                               cls_ids, cls_mask, graph_attn_verb_entity, graph_attn_verb_verb, graph_attn_cls_verb, entity_ids, labels=None)

            if output_mode == "classification":
                loss_fct = CrossEntropyLoss()
                tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
                logit_lst.extend(logits.view(-1, num_labels).tolist())
                label_lst.extend(label_ids.view(-1).tolist())
                input_id_lst.extend(guid_ids.view(-1).tolist())

            elif output_mode == "regression":
                loss_fct = MSELoss()
                tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)

            labels = label_ids.detach().cpu().numpy().tolist()
            start = batch_idx*args.eval_batch_size
            end = start+len(labels)
            batch_range = list(range(start, end))
            csv_names = [eval_examples_table[i].guid.replace("{}-".format(args.plus_test_set), "") for i in batch_range]
            facts = [eval_examples_table[i].fact for i in batch_range]
            labels = label_ids.detach().cpu().numpy().tolist()
            assert len(csv_names) == len(facts) == len(labels)

            temp.extend([(x, y, z) for x, y, z in zip(csv_names, facts, labels)])
            batch_idx += 1

        eval_loss = eval_loss / nb_eval_steps
        preds = preds[0]
        if output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif output_mode == "regression":
            preds = np.squeeze(preds)

        evaluation_results = OrderedDict()
        for x, y, logit in zip(temp, preds, logit_lst):
            c, f, l = x
            if not c in evaluation_results:
                evaluation_results[c] = [{'fact': f, 'gold': int(l), 'pred': int(y), "pred_logits": logit}]
            else:
                evaluation_results[c].append({'fact': f, 'gold': int(l), 'pred': int(y), "pred_logits": logit})

        save_dir = os.path.join(args.output_dir, 'simple-complex-small-test')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        output_eval_file = os.path.join(save_dir, "detailed_{}_test_results.json".format(type))
        with io.open(output_eval_file, "w", encoding='utf-8') as fout:
            json.dump(evaluation_results, fout, sort_keys=True, indent=4)

        result = compute_metrics(preds, all_label_ids.numpy())
        loss = tr_loss/args.period if args.do_train and global_step > 0 else None

        log_step = global_step
        result['eval_loss'] = eval_loss
        result['global_step'] = log_step
        result['loss'] = loss

        output_eval_metrics = os.path.join(save_dir, "{}_test_metrics.txt".format(type))
        with open(output_eval_metrics, "a") as writer:
            logger.info(">>>> {} test results ".format(type))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
    model.train()

def set_tz():
    os.environ['TZ'] = 'US/Eastern'
    time.tzset()


def main():

    set_tz()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    save_code_log_path = args.output_dir
    if not os.path.exists(save_code_log_path):
        os.makedirs(save_code_log_path)

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
                        handlers=[logging.FileHandler("{0}/{1}.log".format(save_code_log_path, 'output')), logging.StreamHandler()])

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    logger.info("\n>> Datasets are loaded from {}\n Outputs will be saved to {}\n".format(args.data_dir, args.output_dir))
    writer = SummaryWriter(os.path.join(args.output_dir, 'events'))

    processor = DataProcessor()
    output_mode = "classification"

    label_list = ["0", "1"]
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples_table = None
    train_examples_verb = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples_table, train_examples_verb = processor.get_examples(args.data_dir, args.data_dir_verb, args.train_set, args.train_set_verb)
        num_train_optimization_steps = int(len(train_examples_table) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
    logger.info(">> The total num of training optimization steps: {}".format(num_train_optimization_steps))

    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))
    model = DoubleBERT(args, tokenizer)
    if args.load_dir:
        logger.info(">> loading the pretrained models from previous ckpt....")
        model.BERT_model_table = BertForSequenceClassification.from_pretrained(args.load_dir, cache_dir=cache_dir, num_labels=2)  # for BCE loss
        model.BERT_model_verb = BertForSequenceClassification.from_pretrained(args.bert_model, cache_dir=args.cache_dir, num_labels=2)
    elif args.load_dir_whole:
        model.load_state_dict(torch.load(args.load_dir_whole + '/full_model.pt'))
    else:
        model.BERT_model_verb = BertForSequenceClassification.from_pretrained(args.bert_model, cache_dir=args.cache_dir,
                                                                          num_labels=2)
        model.BERT_model_table = BertForSequenceClassification.from_pretrained(args.bert_model, cache_dir=args.cache_dir,
                                                                           num_labels=2)
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.do_train:
        param_optimizer = []
        if args.do_freeze_memory:
            for item in list(model.named_parameters()):
                if "BERT_model_verb" not in item[0]:
                    param_optimizer.append(item)
        elif args.do_freeze_table:
            for item in list(model.named_parameters()):
                if "BERT_model_table" not in item[0]:
                    param_optimizer.append(item)
        else:
            param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

        if args.fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.learning_rate,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            if args.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
            warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
                                                 t_total=num_train_optimization_steps)

        else:
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=args.learning_rate,
                                 warmup=args.warmup_proportion,
                                 t_total=num_train_optimization_steps)

    global_step = 0
    tr_loss = 0
    optimizer_flag = 0

    if args.do_train:
        train_features_table, train_features_verb = convert_examples_to_features(
            train_examples_table, train_examples_verb, label_list, args.max_seq_length, tokenizer, fact_place=args.fact, balance=args.balance, phase="train")
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples_table))
        logger.info("  Batch size = %d", args.train_batch_size * args.gradient_accumulation_steps)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        train_dataloader = get_dataloader(train_features_table, train_features_verb, phase="train")

        model.train()
        epoch_count = 0
        best_eval_score = 0.0
        sec_best_eval_score = 0.0

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):   # each epoch
            epoch_count += 1
            logger.info("  Training epoch {} ...".format(epoch))
            nb_tr_examples, nb_tr_steps = 0, 0
            input_id_lst = []
            logit_lst = []
            label_lst = []
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):     # each step (batch)
                batch = tuple(t.to(device) for t in batch)
                input_ids_t, input_mask_t, segment_ids_t, input_ids_v, input_mask_v, segment_ids_v, cls_ids, cls_mask, \
                label_ids, guid_ids, graph_attn_verb_entity, graph_attn_verb_verb, graph_attn_cls_verb, entity_ids = batch

                input_id_lst.extend(guid_ids.tolist())

                logits = model(input_ids_v, segment_ids_v, input_mask_v, input_ids_t, segment_ids_t, input_mask_t,
                               cls_ids, cls_mask, graph_attn_verb_entity, graph_attn_verb_verb, graph_attn_cls_verb, entity_ids, labels=None)

                if output_mode == "classification":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
                    logit_lst.extend(logits.view(-1, num_labels).tolist())
                    label_lst.extend(label_ids.view(-1).tolist())

                elif output_mode == "regression":
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), label_ids.view(-1))

                if n_gpu > 1:
                    loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                writer.add_scalar('train/loss', loss, global_step)
                tr_loss += loss.item()

                nb_tr_examples += input_ids_t.size(0)
                nb_tr_steps += 1

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    total_norm = 0.0
                    for n, p in model.named_parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** (1. / 2)
                    preds = torch.argmax(logits, -1) == label_ids
                    acc = torch.sum(preds).float() / preds.size(0)
                    writer.add_scalar('train/gradient_norm', total_norm, global_step)
                    if args.fp16:
                        lr_this_step = args.learning_rate * warmup_linear.get_lr(global_step, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    model.zero_grad()
                    global_step += 1

                if args.tune_table_after10k:
                    if global_step > 10000 and optimizer_flag == 0:
                        logger.info(">> Start to tune table-bert from 2.5k step....")
                        optimizer_flag = 1
                        param_optimizer = list(model.named_parameters())
                        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
                        optimizer_grouped_parameters = [
                            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                             'weight_decay': 0.01},
                            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                             'weight_decay': 0.0}
                        ]
                        optimizer = BertAdam(optimizer_grouped_parameters,
                                             lr=args.learning_rate,
                                             warmup=args.warmup_proportion,
                                             t_total=num_train_optimization_steps)

                if (step + 1) % args.period == 0:
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_dir = os.path.join(args.output_dir, 'save_step_{}'.format(global_step))

                    model.eval()
                    torch.set_grad_enabled(False)
                    eval_score = evaluate(args, model, device, processor, label_list, num_labels, tokenizer, output_mode, tr_loss,
                             global_step, tbwriter=writer, save_dir=output_dir, save_every_period=False)
                    if eval_score > best_eval_score:
                        save_detailed_test = True
                        if args.do_test:
                            test(args, model, device, processor, label_list, num_labels, tokenizer, output_mode, tr_loss,
                                global_step, tbwriter=writer, save_dir=output_dir, save_detailed_test=save_detailed_test, save_every_period=False)

                    model.train()
                    torch.set_grad_enabled(True)
                    tr_loss = 0

                    if eval_score > best_eval_score or eval_score > sec_best_eval_score:

                        if eval_score > best_eval_score:
                            logger.info("**** update saved_model ****\n >> epoch:{}, eval_res:{:.2}, global_step:{}".format(
                                    epoch_count, best_eval_score, global_step))
                            sec_best_eval_score = best_eval_score
                            best_eval_score = eval_score
                        else:
                            sec_best_eval_score = eval_score

                        output_dir = os.path.join(args.output_dir, 'saved_model_{:.3}k_{:.4}'.format(global_step/1000, best_eval_score))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        # output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
                        # output_config_file = os.path.join(output_dir, CONFIG_NAME)
                        # torch.save(model_to_save.state_dict(), output_model_file)

                        torch.save(model.state_dict(), os.path.join(output_dir, 'full_model.pt'))
                        torch.save(optimizer.state_dict(), os.path.join(output_dir, 'optimizer.pt'))

                        # model_to_save.config.to_json_file(output_config_file)
                        # tokenizer.save_vocabulary(output_dir)

    if args.do_eval or args.do_test:
        if not args.do_train:
            global_step = 0
            output_dir = None
        save_dir = os.path.join(args.output_dir, 'standard_eval-test')
        load_step = args.load_step
        model.eval()
        if args.do_eval:
            evaluate(args, model, device, processor, label_list, num_labels, tokenizer, output_mode, tr_loss,
                 global_step, tbwriter=None, save_dir=save_dir, save_every_period=False)
        if args.do_test:
            test(args, model, device, processor, label_list, num_labels, tokenizer, output_mode, tr_loss,
                 global_step, tbwriter=None, save_dir=save_dir, save_detailed_test=False)
        model.train()

    model.eval()
    if args.do_simple_test:
        other_test("simple", args, model, device, processor, label_list, num_labels, tokenizer, output_mode, tr_loss, global_step)
    if args.do_complex_test:
        other_test("complex", args, model, device, processor, label_list, num_labels, tokenizer, output_mode, tr_loss, global_step)
    if args.do_small_test:
        other_test("small", args, model, device, processor, label_list, num_labels, tokenizer, output_mode, tr_loss, global_step)



if __name__ == "__main__":
    os.environ['TZ'] = 'US/Eastern'
    time.tzset()
    logger.info("Running %s" % ' '.join(sys.argv))

    parser = argparse.ArgumentParser()

    parser.add_argument("--debug", default=False)
    parser.add_argument("--do_train", action='store_true') #
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument("--do_simple_test", action='store_true')
    parser.add_argument("--do_complex_test", action='store_true')
    parser.add_argument("--do_small_test", action='store_true')
    parser.add_argument("--do_freeze_memory", default=False)
    parser.add_argument("--do_freeze_table", default=True)
    parser.add_argument("--tune_table_after10k", default=True)
    parser.add_argument("--max_verb", default=11)
    parser.add_argument("--concat_original_memory", action='store_true')
    parser.add_argument("--load_dir", type=str, help="to accelerate the training process, we can load a table-bert baseline checkpoint")       #
    parser.add_argument("--load_dir_whole", type=str, help="to do eval, load the saved model") #
    parser.add_argument('--load_step', type=int, default=0, help="The checkpoint step to be loaded")
    parser.add_argument("--data_dir", help="same as the input of table-bert") #
    parser.add_argument("--train_set", default="train")
    parser.add_argument("--test_set", default="dev")
    parser.add_argument("--plus_test_set", default="test")
    parser.add_argument("--data_dir_verb", type=str, help="the output of verbalization step")  #
    parser.add_argument("--train_set_verb", default="train_margin")
    parser.add_argument("--test_set_verb", default="dev_margin")
    parser.add_argument("--plus_test_set_verb", default="test_margin")
    parser.add_argument("--num_labels", default=2)
    parser.add_argument("--scan", default="horizontal", choices=["vertical", "horizontal"], type=str, help="The direction of linearizing table cells.")
    parser.add_argument("--output_dir", default='outputs')
    parser.add_argument("--fact", default="second", choices=["first", "second"], type=str, help="Whether to put fact in front.")
    parser.add_argument("--balance", action='store_true', help="balance between + and - samples for training.")
    parser.add_argument("--bert_model", default="bert-base-multilingual-cased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument('--period', type=int, default=3000)
    parser.add_argument("--cache_dir", default="cached_models", type=str, help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--cached_dataset", default="outputs/cached_data/{}".format(time.strftime("%m%d-%H%M%S")), type=str)
    parser.add_argument("--max_seq_length", default=256, type=int)
    parser.add_argument("--max_seq_length_table", default=512, type=int)
    parser.add_argument("--max_seq_length_verb", default=120, type=int)
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--eval_batch_size", default=12, type=int, help="Total batch size for eval.")
    parser.add_argument("--train_batch_size", default=12, type=int, help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=20.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.2, type=float,help="Proportion of training to perform linear learning rate warmup for. ""E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda", action='store_true', help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=6, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true', help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument("--in_dim", default=768)
    parser.add_argument("--mem_dim", default=768)
    parser.add_argument("--clf_in_dim", default=768)
    parser.add_argument("--clf_out_dim", default=1)

    args = parser.parse_args()
    pprint(vars(args))
    sys.stdout.flush()

    main()

