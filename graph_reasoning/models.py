import os
import torch
import tempfile
import tarfile
import torch.nn as nn
from models_BERT import BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert.file_utils import cached_path, WEIGHTS_NAME, CONFIG_NAME
from attention_sigmoid import AttentionLayer_sigmoid
from graph_layer_complex import Graph_Layer
import pdb
import logging
import numpy as np
logger = logging.getLogger(__name__)


class DoubleBERT(nn.Module):
    def __init__(self, args, tokenizer):
        super(DoubleBERT, self).__init__()
        self.args = args
        self.hops = 3

        # BERT
        self.load_dir = args.bert_model
        self.config = self.get_config(self.load_dir, self.args.cache_dir)
        self.tokenizer = tokenizer

        # clf layer
        self.clf_in_dim = 1536
        self.clf_out_dim = 2
        self.in_dim = 768

        # modules
        self.BERT_model_verb = BertForSequenceClassification(self.config, self.args.num_labels)
        self.BERT_model_table = BertForSequenceClassification(self.config, self.args.num_labels)

        self.attention_layer_sigmoid = AttentionLayer_sigmoid(self.args)
        self.dense = nn.Linear(1536, 2, bias=True)
        self.graph_layer = Graph_Layer(self.args)
        self.BERT_model_verb = self.BERT_model_verb.cuda()
        self.BERT_model_table = self.BERT_model_table.cuda()

    def forward(self, input_ids_verb, segment_ids_verb, input_mask_verb,
                input_ids_table, segment_ids_table, input_mask_table, cls_ids, cls_mask,
                graph_attn_verb_entity, graph_attn_verb_verb, graph_attn_cls_verb, entity_ids, labels=None):

        # BERT encoding
        _, last_layer_verb = self.BERT_model_verb(input_ids_verb, segment_ids_verb, input_mask_verb, labels=None)
        logits_table, last_layer_table = self.BERT_model_table(input_ids_table, segment_ids_table, input_mask_table, labels=None)
        cls_table = last_layer_table[:, 0].view(-1, self.in_dim)
        pooled_output = self.BERT_model_table.bert.pooler.dense(cls_table)
        cls_table = self.BERT_model_table.bert.pooler.activation(pooled_output)
        cls_table = self.BERT_model_table.dropout(cls_table)
        cls_table_ = cls_table.unsqueeze(dim=1)

        cls_verbs = last_layer_verb[torch.arange(last_layer_verb.size(0)).unsqueeze(1), cls_ids]
        cls_entity = last_layer_verb[torch.arange(last_layer_verb.size(0)).unsqueeze(1), entity_ids]

        graph_emb = torch.cat((cls_table_, cls_verbs, cls_entity), dim=1)

        # HGCN
        verbs, cls = self.graph_layer(input_emb=graph_emb,
                                 attn_matrix_v_e=graph_attn_verb_entity,
                                 attn_matrix_v_v=graph_attn_verb_verb,
                                 attn_matrix_c_v=graph_attn_cls_verb)

        o = self.attention_layer_sigmoid(cls_table, verbs).view(-1, self.args.mem_dim)

        # Residual
        logits = self.dense(torch.cat((cls_table, o), dim=-1))

        return logits

    def get_config(self, load_dir, cache_dir):
        PRETRAINED_MODEL_ARCHIVE_MAP = {
            'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
            'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
            'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
            'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
            'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
            'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
            'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
        }
        BERT_CONFIG_NAME = 'bert_config.json'
        archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[load_dir]
        resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)

        if resolved_archive_file == archive_file:
            logger.info("loading archive file {}".format(archive_file))
        else:
            logger.info("loading archive file {} from cache at {}".format(archive_file, resolved_archive_file))
        if os.path.isdir(resolved_archive_file):
            serialization_dir = resolved_archive_file
        else:
            tempdir = tempfile.mkdtemp()
            logger.info("extracting archive file {} to temp dir {}".format(
                resolved_archive_file, tempdir))
            with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                archive.extractall(tempdir)
            serialization_dir = tempdir
        config_file = os.path.join(serialization_dir, CONFIG_NAME)
        if not os.path.exists(config_file):
            config_file = os.path.join(serialization_dir, BERT_CONFIG_NAME)
        config = BertConfig.from_json_file(config_file)
        return config
