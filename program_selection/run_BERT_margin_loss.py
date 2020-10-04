# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import random
import sys
import io
import json
import numpy as np
import torch
import time
import shutil
from pprint import pprint
import pickle as pkl
import csv
import re
import random
import logging
from tqdm import tqdm, trange
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from modeling import BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
from tensorboardX import SummaryWriter
from collections import OrderedDict

logger = logging.getLogger(__name__)

entity_linking_pattern = re.compile('#.*?;-*[0-9]+,(-*[0-9]+)#')  # match #a,b;-1,-1# and replace it with ENTITY
fact_pattern = re.compile('#(.*?);-*[0-9]+,-*[0-9]+#')
unk_pattern = re.compile('#([^#]+);-1,-1#')
TSV_DELIM = "\t"
TBL_DELIM = " ; "


def parse_fact(fact):
    fact = re.sub(unk_pattern, '[UNK]', fact)  # optional
    chunks = re.split(fact_pattern, fact)
    output = ' '.join([x.strip() for x in chunks if len(x.strip()) > 0])
    return output


class InputExample(object):
    def __init__(self, guid, table_name, text_a, text_b, label, pred_label=None, true_label=None):
        ''' 
        Args:
            guid:   unique id
            table_name: name of table
            text_a: statement
            text_b: program
            label:  positive / negative
            pred_label: program label
            true_label: statement lable
        '''
        self.guid = guid
        self.table_name = table_name
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.pred_label = pred_label
        self.true_label = true_label


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id, guid, pred_label, true_label):
        self.guid = guid
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.pred_label = pred_label
        self.true_label = true_label


class LpaProcessor(object):
    def get_examples(self, data_dir, dataset=None):
        logger.info('Get examples from: {}.tsv'.format(dataset))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "{}.tsv".format(dataset))))

    def get_labels(self):
        return [0, 1], len([0, 1])

    def _read_tsv(cls, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for idx, line in enumerate(f):
                entry = line.strip().split('\t')
                index = int(entry[1].split('-')[-1])
                table_name = entry[0]
                true_label = int(entry[2])  # stat label
                pred_label = int(entry[3])  # prog label
                statement = parse_fact(entry[4])
                program = entry[5]
                label = int(entry[6])
                lines.append([index, statement, program, label, true_label, pred_label, table_name])

            return lines

    def _create_examples(self, lines):
        examples = []
        tmp_map = {}
        for (i, line) in enumerate(lines):
            guid = line[0]
            table_name = line[6]
            text_a = line[1]
            text_b = line[2]
            label = line[3]
            true_label = line[4]
            pred_label = line[5]
            if guid not in tmp_map:
                tmp_map[guid] = []
            tmp_map[guid].append(InputExample(guid=guid, table_name=table_name, text_a=text_a, text_b=text_b, label=label,
                                              pred_label=pred_label, true_label=true_label))
        for item in tmp_map:
            examples.append(tmp_map[item])

        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []

    for (ex_index, combined_example) in enumerate(tqdm(examples, desc="convert to features")):
        tmp_features = []
        for example in combined_example:
            guid = example.guid
            pred_label = example.pred_label
            true_label = example.true_label
            label_id = label_map[example.label]

            tokens_a = tokenizer.tokenize(example.text_a)
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * (len(tokens_a) + 2)
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            # if ex_index < 1:
            #     logger.info("*** Example ***")
            #     logger.info("guid: %s" % (example.guid))
            #     logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            #     logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            #     logger.info("label: %s (id = %d, pred_label = %d, fact_label = %d)" % (
            #     example.label, label_id, pred_label, true_label))

            tmp_features.append(InputFeatures(input_ids=input_ids,
                                              input_mask=input_mask,
                                              segment_ids=segment_ids,
                                              label_id=label_id,
                                              guid=guid,
                                              pred_label=pred_label,
                                              true_label=true_label))
        features.append(tmp_features)

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def eval_2(preds):
    num = len(preds)
    success = preds.sum()
    fail = num - success
    acc = success / (success + fail + 0.001)
    return success, fail, acc


def compute_metrics(preds):
    success, fail, acc = eval_2(preds)
    result = {"success": success, "fail": fail, "acc": acc}
    return result


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_dataLoader(args, processor, tokenizer, phase=None):
    dataset_dict = {"train": args.train_set, "dev": args.dev_set, "std_test": args.std_test_set,
                    "complex_test": args.complex_test_set,
                    "small_test": args.small_test_set, "simple_test": args.simple_test_set}

    label_list, _ = processor.get_labels()
    examples = processor.get_examples(args.data_dir, dataset_dict[phase])
    features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer)

    batch_size = args.train_batch_size if phase == "train" else args.eval_batch_size
    epoch_num = args.num_train_epochs if phase == "train" else 1

    num_optimization_steps = int(len(examples) / batch_size / args.gradient_accumulation_steps) * epoch_num
    logger.info("Examples#: {}, Batch size: {}".format(len(examples), batch_size * args.gradient_accumulation_steps))
    logger.info("Total num of steps#: {}, Total num of epoch#: {}".format(num_optimization_steps, epoch_num))

    all_input_ids, all_input_mask, all_segment_ids, all_guid, all_label_ids, \
    all_pred_label, all_true_label = [], [], [], [], [], [], []
    for combined_features in features:
        tmp_input_ids, tmp_input_mask, tmp_segment_ids, tmp_guid, tmp_label_ids, \
        tmp_pred_label, tmp_true_label = [], [], [], [], [], [], []
        for f in combined_features:
            tmp_input_ids.append(f.input_ids)
            tmp_input_mask.append(f.input_mask)
            tmp_segment_ids.append(f.segment_ids)
            tmp_guid.append(f.guid)
            tmp_label_ids.append(f.label_id)
            tmp_pred_label.append(f.pred_label)
            tmp_true_label.append(f.true_label)
        all_input_ids.append(torch.tensor(tmp_input_ids, dtype=torch.long))
        all_input_mask.append(torch.tensor(tmp_input_mask, dtype=torch.long))
        all_segment_ids.append(torch.tensor(tmp_segment_ids, dtype=torch.long))
        all_guid.append(torch.tensor(tmp_guid, dtype=torch.long))
        all_label_ids.append(torch.tensor(tmp_label_ids, dtype=torch.long))
        all_pred_label.append(torch.tensor(tmp_pred_label, dtype=torch.long))
        all_true_label.append(torch.tensor(tmp_true_label, dtype=torch.long))

    dataloader = []
    index = [i for i in range(len(all_input_ids))]
    if phase != "train" or args.do_gen:
        for idx in index:
            dataloader.append((all_input_ids[idx], all_input_mask[idx], all_segment_ids[idx], all_guid[idx],
                               all_label_ids[idx], all_pred_label[idx], all_true_label[idx]))
    else:
        random.shuffle(index)
        random.shuffle(index)
        for idx in index:
            # set limit to program number for the CUDA memory limitation
            prog_limit = min(args.max_prog_num, len(all_input_ids[idx]))
            dataloader.append(
                (all_input_ids[idx][:prog_limit], all_input_mask[idx][:prog_limit], all_segment_ids[idx][:prog_limit],
                 all_guid[idx][:prog_limit], all_label_ids[idx][:prog_limit], all_pred_label[idx][:prog_limit],
                 all_true_label[idx][:prog_limit]))

    return dataloader, num_optimization_steps, examples


def save_model(model_to_save, tokenizer):
    save_model_dir = os.path.join(args.output_dir, 'saved_model')
    mkdir(save_model_dir)
    output_model_file = os.path.join(save_model_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(save_model_dir, CONFIG_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(save_model_dir)


def margin_max(logits, label_ids):
    softmax_logits = torch.softmax(logits.view(-1), dim=0)
    pos_max_logit = torch.max(softmax_logits * label_ids)
    neg_max_logit = torch.max(softmax_logits * (1 - label_ids))
    loss = torch.nn.functional.relu(0.15 + neg_max_logit - pos_max_logit)
    return loss


def run_train(device, processor, tokenizer, model, writer, phase="train"):
    tr_dataloader, tr_num_steps, tr_examples = get_dataLoader(args, processor, tokenizer, phase=phase)

    model.train()

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = \
        [{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
         {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate,
                         warmup=args.warmup_proportion, t_total=tr_num_steps)
    optimizer.zero_grad()

    global_step = 0
    best_acc = 0.0
    sum_loss = 0.0
    n_gpu = torch.cuda.device_count()

    for ep in trange(args.num_train_epochs, desc="Training"):
        for step, batch in enumerate(tqdm(tr_dataloader)):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, guid, label_ids, pred_label, true_label = batch
            if 1 in label_ids.tolist() and input_ids.size()[0] != 1:
                logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=None)
                loss = margin_max(logits, label_ids)
            else:
                continue

            if n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            sum_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:  # optimizer
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                writer.add_scalar('{}/loss'.format(phase), (sum_loss / global_step / args.gradient_accumulation_steps),
                                  global_step)

                if model.training is False:
                    print("The mode is wrong during training...")
                    exit(-1)

            model.eval()
            torch.set_grad_enabled(False)

            if args.do_eval and step % args.period == 0:
                model_to_save = model.module if hasattr(model, 'module') else model

                dev_acc = run_eval(device, processor, tokenizer, model, writer, global_step, tensorboard=True,
                                   phase="dev")

                if dev_acc > best_acc:
                    best_acc = dev_acc
                    logger.info(">> Save model. Best acc: {:.4}. Epoch {}".format(best_acc, ep))
                    save_model(model_to_save, tokenizer)  # save model
                logger.info(">> Now the best acc is {:.4}\n".format(best_acc))
                # run_eval(device, processor, tokenizer, model, writer, global_step, save_detailed_res, tensorboard=True, phase="std_test")
            model.train()
            torch.set_grad_enabled(True)

    return global_step


def run_eval(device, processor, tokenizer, model, writer, global_step, save_detailed_res=False, tensorboard=False,
             phase=None):
    sys.stdout.flush()
    logger.info("\n************ Start {} *************".format(phase))

    model.eval()

    dataloader, num_steps, examples = get_dataLoader(args, processor, tokenizer, phase=phase)

    eval_loss = 0.0
    num_steps = 0
    preds = []

    for step, batch in enumerate(tqdm(dataloader, desc=phase)):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, guid_ids, label_ids, pred_label, true_label = batch
        assert len(input_ids) == len(input_mask) == len(segment_ids) == len(guid_ids) == len(label_ids) == len(
            pred_label) == len(true_label)
        num_steps += 1

        with torch.no_grad():
            logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=None)

            max_index = np.argmax(torch.softmax(logits.view(-1), dim=0).tolist())
            pred_one = pred_label[max_index]
            true_one = true_label[max_index]
            prediction_res = int(pred_one == true_one)
            preds.append(prediction_res)

            eval_loss += margin_max(logits, label_ids)

    eval_loss /= num_steps

    result = compute_metrics(np.asarray(preds))
    result['{}_loss'.format(phase)] = eval_loss
    result['global_step'] = global_step
    logger.info(result)

    if tensorboard and writer is not None:
        for key in sorted(result.keys()):
            writer.add_scalar('{}/{}'.format(phase, key), result[key], global_step)

    model.train()

    return result['acc']


def evalSelectMostProb(device, processor, tokenizer, model, writer, global_step, save_detailed_res=False,
                       tensorboard=False, phase=None):
    sys.stdout.flush()
    logger.info("*** start phase {} ***".format(phase))
    save_path = os.path.join(args.output_dir, "{}_program_select_only_true.json".format(phase))

    model.eval()

    dataloader, num_steps, examples = get_dataLoader(args, processor, tokenizer, phase=phase)

    eval_loss = 0.0
    num_steps = 0
    preds = []
    all_labels = []
    mapping = OrderedDict()
    start = 0

    for step, batch in enumerate(tqdm(dataloader, desc=phase)):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, guid_ids, label_ids, pred_label, true_label = batch
        assert len(input_ids) == len(input_mask) == len(segment_ids) == len(guid_ids) == len(label_ids) == len(
            pred_label) == len(true_label)
        num_steps += 1

        with torch.no_grad():
            logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=None)
            max_index = np.argmax(torch.softmax(logits.view(-1), dim=0).tolist())
            pred_one = pred_label[max_index]
            true_one = true_label[max_index]
            prediction_res = int(pred_one == true_one)
            preds.append(prediction_res)

            eval_loss += margin_max(logits, label_ids)

            end = start + len(input_ids)
            batch_range = list(range(start, end))
            start += len(input_ids)

            table_names = [item.table_name for item in examples[step]]
            ids = [item.guid for item in examples[step]]
            t_a = [item.text_a for item in examples[step]]
            t_b = [item.text_b for item in examples[step]]
            labels = label_ids.detach().cpu().numpy().tolist()
            fact_lables = true_label.detach().cpu().numpy().tolist()
            prog_lables = pred_label.detach().cpu().numpy().tolist()

            assert len(ids) == len(t_a) == len(t_b) == len(labels) == len(prog_lables) == len(fact_lables) == len(
                table_names)

            original_data = [
                str(t_n) + '\t' + 'nt-' + str(i) + '\t' + str(tl) + '\t' + str(pl) + '\t' + ta + '\t' + tb + '\t' + str(
                    gl)
                for t_n, i, ta, tb, gl, tl, pl in zip(table_names, ids, t_a, t_b, labels, fact_lables, prog_lables)]

            logits_sigmoid = torch.sigmoid(logits).view(-1)
            similarity = logits_sigmoid.detach().cpu().numpy()
            for i, s, p, f, l, ori_data in zip(ids, similarity, prog_lables, fact_lables, labels, original_data):
                if i not in mapping.keys():
                    mapping[i] = [s, p, f, l, ori_data]  # [simi, prog_label, fact_label, gold_label]
                else:
                    if s > mapping[i][0]:
                        mapping[i] = [s, p, f, l, ori_data]

    eval_loss /= num_steps

    result = compute_metrics(np.asarray(preds))
    result['{}_loss'.format(phase)] = eval_loss
    result['global_step'] = global_step
    logger.info(result)

    # --- save detailed results ---
    results = []
    success, fail = 0, 0
    logger.info("----mapping:{}".format(len(mapping.items())))
    for i, line in mapping.items():
        if line[1] == line[2]:
            success += 1
        else:
            fail += 1
        results.append({'similarity': str(line[0]), 'originaldata': line[4]})
    with open(save_path, 'w', encoding='utf-8') as fout:
        json.dump(results, fout, indent=2)

    if tensorboard and writer is not None:
        for key in sorted(result.keys()):
            writer.add_scalar('{}/{}'.format(phase, key), result[key], global_step)

    model.train()

    return result['acc']


def main():
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    writer = SummaryWriter(os.path.join(args.output_dir, 'events'))
    cache_dir = args.cache_dir

    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    save_code_log_path = args.output_dir
    mkdir(args.output_dir)

    logging.basicConfig(format='%(message)s', datefmt='%m/%d/%Y %H:%M', level=logging.INFO,
                        handlers=[logging.FileHandler("{0}/{1}.log".format(save_code_log_path, 'output')),
                                  logging.StreamHandler()])
    logger.info(args)
    logger.info("Command is: %s" % ' '.join(sys.argv))
    logger.info("Device: {}, n_GPU: {}".format(device, n_gpu))
    logger.info("Datasets are loaded from {}\nOutputs will be saved to {}\n".format(args.data_dir, args.output_dir))

    processor = LpaProcessor()

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    load_dir = args.load_dir if args.load_dir else args.bert_model
    logger.info('Model is loaded from %s' % load_dir)

    model = BertForSequenceClassification.from_pretrained(load_dir, cache_dir=cache_dir, num_labels=args.num_labels)
    model.to(device)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.do_train:
        logger.info("\n************ Start Training *************")
        run_train(device, processor, tokenizer, model, writer, phase="train")

    if args.do_eval:
        run_eval(device, processor, tokenizer, model, writer, global_step=0, save_detailed_res=True, tensorboard=False,
                 phase="dev")
        run_eval(device, processor, tokenizer, model, writer, global_step=0, save_detailed_res=True, tensorboard=False,
                 phase="std_test")

    if args.do_complex_test:
        run_eval(device, processor, tokenizer, model, writer, global_step=0, save_detailed_res=True, tensorboard=False,
                 phase="complex_test")

    if args.do_small_test:
        run_eval(device, processor, tokenizer, model, writer, global_step=0, save_detailed_res=True, tensorboard=False,
                 phase="small_test")

    if args.do_simple_test:
        run_eval(device, processor, tokenizer, model, writer, global_step=0, save_detailed_res=True, tensorboard=False,
                 phase="simple_test")

    if args.do_gen:
        evalSelectMostProb(device, processor, tokenizer, model, writer, global_step=0, save_detailed_res=True,
                           tensorboard=False, phase="small_test")
        evalSelectMostProb(device, processor, tokenizer, model, writer, global_step=0, save_detailed_res=True,
                           tensorboard=False, phase="train")
        evalSelectMostProb(device, processor, tokenizer, model, writer, global_step=0, save_detailed_res=True,
                           tensorboard=False, phase="dev")
        evalSelectMostProb(device, processor, tokenizer, model, writer, global_step=0, save_detailed_res=True,
                           tensorboard=False, phase="std_test")
        evalSelectMostProb(device, processor, tokenizer, model, writer, global_step=0, save_detailed_res=True,
                           tensorboard=False, phase="simple_test")
        evalSelectMostProb(device, processor, tokenizer, model, writer, global_step=0, save_detailed_res=True,
                           tensorboard=False, phase="complex_test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--do_simple_test", action='store_true')
    parser.add_argument("--do_complex_test", action='store_true')
    parser.add_argument("--do_small_test", action='store_true')
    parser.add_argument("--do_gen", action='store_true')
    parser.add_argument("--max_prog_num", default=47)  # set to 50 when enough CUDA memory is available
    parser.add_argument("--num_labels", default=1)
    parser.add_argument("--load_dir", help="path to model checkpoints")
    parser.add_argument("--data_dir", help="path to data")
    parser.add_argument("--train_set", default="train")
    parser.add_argument("--dev_set", default="dev")
    parser.add_argument("--std_test_set", default="test")
    parser.add_argument("--small_test_set", default="small_test")
    parser.add_argument("--complex_test_set", default="complex_test")
    parser.add_argument("--simple_test_set", default="simple_test")
    parser.add_argument("--output_dir", default='outputs-gen', help="dir for logs and checkpoints")
    parser.add_argument("--cache_dir", default="cached_models", type=str, help="save downloaded pre-trained models")
    parser.add_argument('--period', type=int, default=1000)
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="list: bert-base-uncased, bert-large-uncased, bert-base-cased, bert-large-cased, "
                             "bert-base-multilingual-uncased, bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--do_lower_case", default=True, help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", default=128)
    parser.add_argument("--train_batch_size", default=4)
    parser.add_argument("--eval_batch_size", default=4)
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=20)
    parser.add_argument("--warmup_proportion", default=0.3, type=float, help="0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42, help="random seed")

    args = parser.parse_args()
    main()


