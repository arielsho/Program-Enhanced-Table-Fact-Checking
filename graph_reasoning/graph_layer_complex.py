import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import sys
import math
import logging
import tempfile
import tarfile
import numpy as np
import os
logger = logging.getLogger(__name__)
from models_BERT import BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert.file_utils import cached_path, WEIGHTS_NAME, CONFIG_NAME


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu}

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
    class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        # pdb.set_trace()
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = 3
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs.float(), value_layer.float())
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

def get_config(load_dir, cache_dir):
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


class Graph_Layer(nn.Module):
    def __init__(self, args):
        super(Graph_Layer, self).__init__()
        self.args = args
        self.load_dir = args.bert_model
        self.config = get_config(self.load_dir, self.args.cache_dir)
        self.gcn_layer = BertLayer(self.config)

    def forward(self, input_emb, attn_matrix_v_e, attn_matrix_v_v, attn_matrix_c_v):
        attn_matrix_v_e = torch.unsqueeze(- (1 - attn_matrix_v_e) * 1e06, dim=1)
        attn_matrix_v_v = torch.unsqueeze(- (1 - attn_matrix_v_v) * 1e06, dim=1)
        attn_matrix_c_v = torch.unsqueeze(- (1 - attn_matrix_c_v) * 1e06, dim=1)
        attn_matrix = torch.cat((attn_matrix_v_e, attn_matrix_v_v, attn_matrix_c_v), dim=1)
        output = self.gcn_layer(input_emb, attn_matrix)
        output_verbs = output[:, 1:12, :]
        output_cls = output[:, 0, :]

        return output_verbs, output_cls
