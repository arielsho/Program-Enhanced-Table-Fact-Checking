import json
import re
import pandas
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
from pprint import pprint
from APIs import *
from tqdm import tqdm
from collections import Counter, OrderedDict
import argparse
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import logging
import re
import sys

entity_linking_pattern = re.compile('#.*?;-*[0-9]+,(-*[0-9]+)#')
fact_pattern = re.compile('#(.*?);-*[0-9]+,-*[0-9]+#')
unk_pattern = re.compile('#([^#]+);-1,-1#')
TSV_DELIM = "\t"
TBL_DELIM = " ; "


def parse_fact(fact):
    fact = re.sub(unk_pattern, '[UNK]', fact)
    chunks = re.split(fact_pattern, fact)
    output = ' '.join([x.strip() for x in chunks if len(x.strip()) > 0])
    return output


operators = ['count', 'inc_num', 'dec_num', 'within_s_s', 'within_n_n',
             'not_within_s_s', 'not_within_n_n', 'none', 'only', 'several', 'zero', 'after', 'before',
             'top', 'bottom', 'first', 'second', 'third', 'fourth', 'fifth', 'last', 'uniq_num', 'uniq_str',
             'avg', 'sum', 'max', 'min', 'argmax', 'argmin', 'str_hop', 'most_freq', 'num_hop', 'half',
             'one_third', 'diff', 'add', 'greater', 'less', 'eq', 'not_eq', 'str_eq', 'not_str_eq', 'and',
             'filter_str_eq', 'filter_str_not_eq', 'filter_eq', 'filter_not_eq', 'filter_less', 'filter_greater',
             'filter_greater_eq', 'filter_less_eq', 'all_str_eq', 'all_str_not_eq', 'all_eq', 'all_not_eq',
             'all_less', 'all_less_eq', 'all_greater', 'all_greater_eq', 'hop', 'uniq', 'not_within', 'within']


def isnumber(string):
    return string in [numpy.dtype('int64'), numpy.dtype('int32'), numpy.dtype('float32'), numpy.dtype('float64')]


class MyNode():
    def __init__(self, val):
        self.args = []
        self.val = val
        self.temp = ""
        self.id = -1


class ExpressionTree(object):
    def buildTree(self, expression, st, ed0):
        if expression is ['no program']:
            return None
        if len(expression) == 0:
            return None
        root_val = expression[st]
        root = MyNode(root_val)

        ed = self.findIndex(expression, st + 1, ed0)  # last index in expression lst

        # parse the arguments
        cur = []
        stack = []
        pair = []
        i = st + 1
        while i < ed + 1:
            if expression[i] == '{' or expression[i] == '}':
                if expression[i] == '{':
                    stack.append(i)
                else:
                    if len(stack) == 0:
                        print('invalid')
                        return None
                    pair.append((stack[-1], i))
                    stack.pop(-1)
                i += 1
                continue
            if expression[i] != ';':
                if expression[i] not in operators:
                    cur.append(expression[i])  # add variables
                    i += 1
                else:
                    if i < ed:
                        if expression[i + 1] != '{':
                            cur.append(expression[i])  # add variables
                            i += 1
                            continue

                    cur_new_ed = self.findIndex(expression, i, ed + 1)
                    cur.append(self.buildTree(expression, i, cur_new_ed + 1))  # build sub-tree: cur_op_node {....}
                    i = cur_new_ed + 1
            else:
                if len(cur) != 0:
                    if type(cur[0]) == str:
                        root.args.append(MyNode(' '.join(cur)))
                    else:
                        root.args.append(cur[0])
                    cur = []
                i += 1
        if len(cur) != 0:
            if type(cur[0]) == str:
                root.args.append(MyNode(' '.join(cur)))
            else:
                root.args.append(cur[0])

        return root

    def execute(self, func, args, arg_num):
        if func is None or arg_num == 0:
            return None
        if func in APIs and len(args) >= arg_num:
            try:
                return APIs[func]['function'](*args[:arg_num])
            except:
                return None
        else:
            return None

    def convertAns2Template(self, ops, arg, ans, temp_arg1, temp_arg2, temp_arg3, column_name):
        if ops == 'none':
            if ans:
                return '*' + str(arg[0]) + ' is none value', str(arg[0]) + ' is none value'
            else:
                return '*' + str(arg[0]) + ' is not none value', str(arg[0]) + ' is not none value'
        if ops == 'eq' or ops == 'str_eq':
            if ans:
                return '*' + str(arg[1]) + ' is equal to ' + '*' + str(arg[0]), str(arg[1]) + ' is equal to ' + str(
                    arg[0])
            else:
                return '*' + str(arg[1]) + ' is not equal to ' + '*' + str(arg[0]), str(
                    arg[1]) + ' is not equal to ' + str(arg[0])
        if ops == 'not_eq' or ops == 'not_str_eq':
            if ans:
                return '*' + str(arg[1]) + ' is not equal to ' + '*' + str(arg[0]), str(
                    arg[1]) + ' is not equal to ' + str(arg[0])
            else:
                return '*' + str(arg[1]) + ' is equal to ' + '*' + str(arg[0]), str(arg[1]) + ' is equal to ' + str(
                    arg[0])

        if ops == 'count':
            return 'the amount ' + '*' + temp_arg1 + ' is ' + '*' + str(ans), 'the amount ' + temp_arg1

        if ops == 'within_s_s':
            if ans:
                return '*' + str(arg[2]) + ' in ' + '*' + temp_arg1 + ' where column ' + '*' + str(arg[1]), str(
                    arg[2]) + ' in ' + temp_arg1 + ' where column ' + str(arg[1])
            else:
                return '*' + str(arg[2]) + ' not in ' + '*' + temp_arg1 + ' where column ' + '*' + str(arg[1]), str(
                    arg[2]) + ' not in ' + temp_arg1 + ' where column ' + str(arg[1])

        if ops == 'within_n_n':
            if ans:
                return '*' + str(arg[2]) + ' in ' + '*' + temp_arg1 + ' where column ' + '*' + str(arg[1]), str(
                    arg[2]) + ' in ' + temp_arg1 + ' where column ' + str(arg[1])
            else:
                return '*' + str(arg[2]) + ' not in ' + '*' + temp_arg1 + ' where column ' + '*' + str(arg[1]), str(
                    arg[2]) + ' not in ' + temp_arg1 + ' where column ' + str(arg[1])

        if ops == 'not_within_s_s':
            if ans:
                return '*' + str(arg[2]) + ' not in ' + '*' + temp_arg1 + ' where column ' + '*' + str(arg[1]), str(
                    arg[2]) + ' not in ' + temp_arg1 + ' where column ' + str(arg[1])
            else:
                return '*' + str(arg[2]) + ' in ' + '*' + temp_arg1 + ' where column ' + '*' + str(arg[1]), str(
                    arg[2]) + ' in ' + temp_arg1 + ' where column ' + str(arg[1])

        if ops == 'not_within_n_n':
            if ans:
                return '*' + str(arg[2]) + ' not in ' + '*' + temp_arg1 + ' where column ' + '*' + str(arg[1]), str(
                    arg[2]) + ' not in ' + temp_arg1 + ' where column ' + str(arg[1])
            else:
                return '*' + str(arg[2]) + ' in ' + '*' + temp_arg1 + ' where column ' + '*' + str(arg[1]), str(
                    arg[2]) + ' in ' + temp_arg1 + ' where column ' + str(arg[1])

        if ops == 'only':
            cur_idxs = arg[0].index
            table_idxs = []
            for item in cur_idxs:
                table_idxs.append('row ' + str(item + 1))
            if ans:
                return '*' + temp_arg1 + ' is only one', 'only ' + temp_arg1
            else:
                return '*' + temp_arg1 + ' is not only one', 'not only ' + temp_arg1

        if ops == 'several':
            cur_idxs = arg[0].index
            table_idxs = []
            for item in cur_idxs:
                table_idxs.append('row ' + str(item + 1))
            if ans:
                return '*' + temp_arg1 + ' is more than 1', 'several ' + temp_arg1
            else:
                return '*' + temp_arg1 + ' is less or equal to 1', 'less or equal than 1 ' + temp_arg1

        if ops == 'zero':
            if ans:
                return '*' + temp_arg1 + ' not exist', temp_arg1 + ' not exist'
                # return str(arg[0]) + ' is 0'
            else:
                return '*' + temp_arg1 + ' exist', temp_arg1 + ' exist'

        if ops == 'after':
            if ans:
                return '*' + temp_arg1 + ' after ' + '*' + temp_arg2, temp_arg1 + ' after ' + temp_arg2
            else:
                return '*' + temp_arg1 + ' before ' + '*' + temp_arg2, temp_arg1 + ' before ' + temp_arg2

        if ops == 'before':
            if ans:
                return '*' + temp_arg1 + ' before ' + '*' + temp_arg2, temp_arg1 + ' before ' + temp_arg2
            else:
                return '*' + temp_arg2 + ' after ' + '*' + temp_arg1, temp_arg2 + ' after ' + temp_arg1

        if ops == 'top':
            cur_idx = ans.index
            cur_idxs = arg[0].index
            table_idxs = []
            for item in cur_idxs:
                table_idxs.append('row ' + str(item + 1))
            return 'the first ' + '*' + temp_arg1 + ' is row ' + '*' + str(cur_idx[0]), 'the first ' + temp_arg1

        if ops == 'bottom':
            cur_idx = ans.index
            cur_idxs = arg[0].index
            table_idxs = []
            for item in cur_idxs:
                table_idxs.append('row ' + str(item + 1))
            return 'the last ' + '*' + temp_arg1 + ' is row ' + '*' + str(cur_idx[0]), 'the last ' + temp_arg1

        if ops == 'first':
            if len(arg[0]) == 1:
                arg[0], arg[1] = arg[1], arg[0]
            cur_idx = arg[1].index
            cur_idxs = arg[0].index
            table_idxs = []
            for item in cur_idxs:
                table_idxs.append('row ' + str(item + 1))
            if ans:
                return '*' + temp_arg2 + ' of ' + '*' + temp_arg1 + ' is first', temp_arg2 + ' of ' + temp_arg1 + ' is first'
            else:
                return '*' + temp_arg2 + ' of ' + '*' + temp_arg1 + ' is not first', temp_arg2 + ' of ' + temp_arg1 + ' is not first'

        if ops == 'second':
            if len(arg[0]) == 1:
                arg[0], arg[1] = arg[1], arg[0]
            cur_idx = arg[1].index
            cur_idxs = arg[0].index
            table_idxs = []
            for item in cur_idxs:
                table_idxs.append('row ' + str(item + 1))
            if ans:
                return '*' + temp_arg2 + ' of ' + '*' + temp_arg1 + ' is second', temp_arg2 + ' of ' + temp_arg1 + ' is second'
            else:
                return '*' + temp_arg2 + ' of ' + '*' + temp_arg1 + ' is not second', temp_arg2 + ' of ' + temp_arg1 + ' is not second'
        if ops == 'third':
            if len(arg[0]) == 1:
                arg[0], arg[1] = arg[1], arg[0]
            cur_idx = arg[1].index
            cur_idxs = arg[0].index
            table_idxs = []
            for item in cur_idxs:
                table_idxs.append('row ' + str(item + 1))
            if ans:
                return '*' + temp_arg2 + ' of ' + '*' + temp_arg1 + ' is third', temp_arg2 + ' of ' + temp_arg1 + ' is third'
            else:
                return '*' + temp_arg2 + ' of ' + '*' + temp_arg1 + ' is not third', temp_arg2 + ' of ' + temp_arg1 + ' is not third'

        if ops == 'fourth':
            if len(arg[0]) == 1:
                arg[0], arg[1] = arg[1], arg[0]
            cur_idx = arg[1].index
            cur_idxs = arg[0].index
            table_idxs = []
            for item in cur_idxs:
                table_idxs.append('row ' + str(item + 1))
            if ans:
                return '*' + temp_arg2 + ' of ' + '*' + temp_arg1 + ' is fourth', temp_arg2 + ' of ' + temp_arg1 + ' is fourth'
            else:
                return '*' + temp_arg2 + ' of ' + '*' + temp_arg1 + ' is not fourth', temp_arg2 + ' of ' + temp_arg1 + ' is not fourth'

        if ops == 'fifth':
            if len(arg[0]) == 1:
                arg[0], arg[1] = arg[1], arg[0]
            cur_idx = arg[1].index
            cur_idxs = arg[0].index
            table_idxs = []
            for item in cur_idxs:
                table_idxs.append('row ' + str(item + 1))
            if ans:
                return '*' + temp_arg2 + ' of ' + '*' + temp_arg1 + ' is fifth', temp_arg2 + ' of ' + temp_arg1 + ' is fifth'
            else:
                return '*' + temp_arg2 + ' of ' + '*' + temp_arg1 + ' is not fifth', temp_arg2 + ' of ' + temp_arg1 + ' is not fifth'
        if ops == 'last':
            if len(arg[0]) == 1:
                arg[0], arg[1] = arg[1], arg[0]
            cur_idx = arg[1].index
            cur_idxs = arg[0].index
            table_idxs = []
            for item in cur_idxs:
                table_idxs.append('row ' + str(item + 1))
            if ans:
                return '*' + temp_arg2 + ' of ' + '*' + temp_arg1 + ' is last', temp_arg2 + ' of ' + temp_arg1 + ' is last'
            else:
                return '*' + temp_arg2 + ' of ' + '*' + temp_arg1 + ' is not last', temp_arg2 + ' of ' + temp_arg1 + ' is not last'

        if ops == 'uniq_num' or ops == 'uniq_str':
            cur_idxs = arg[0].index
            table_idxs = []
            for item in cur_idxs:
                table_idxs.append('row ' + str(item + 1))
            return 'the unique value in ' + '*' + temp_arg1 + ' is ' + '*' + str(
                ans), 'the unique value in ' + temp_arg1

        if ops == 'avg':
            cur_idxs = arg[0].index
            table_idxs = []
            for item in cur_idxs:
                table_idxs.append('row ' + str(item + 1))
            return 'the average value of ' + '*' + temp_arg1 + ' is ' + '*' + str(
                ans), 'the average value of ' + temp_arg1

        if ops == 'min':
            cur_idxs = arg[0].index
            table_idxs = []
            for item in cur_idxs:
                table_idxs.append('row ' + str(item + 1))
            condi = temp_arg1 if temp_arg1 == '' else ' in ' + temp_arg1
            return 'the min value of column ' + '*' + str(arg[1]) + condi + ' is ' + '*' + str(
                ans), 'the min value of column ' + str(arg[1]) + condi

        if ops == 'max':
            cur_idxs = arg[0].index
            table_idxs = []
            for item in cur_idxs:
                table_idxs.append('row ' + str(item + 1))
            condi = temp_arg1 if temp_arg1 == '' else ' in ' + temp_arg1
            return 'the max value of column ' + '*' + str(arg[1]) + condi + ' is ' + '*' + str(
                ans), 'the max value of column ' + str(arg[1]) + condi

        if ops == 'sum':
            cur_idxs = arg[0].index
            table_idxs = []
            for item in cur_idxs:
                table_idxs.append('row ' + str(item + 1))
            condi = temp_arg1 if temp_arg1 == '' else ' in ' + temp_arg1
            return 'the sum value of column ' + '*' + str(arg[1]) + condi + ' is ' + '*' + str(
                ans), 'the sum value of column ' + str(arg[1]) + condi

        if ops == 'argmax':
            cur_idxs = arg[0].index
            table_idxs = []
            for item in cur_idxs:
                table_idxs.append('row ' + str(item + 1))
            condi = temp_arg1 if temp_arg1 == '' else ' in ' + temp_arg1
            return 'the row with max value in column ' + '*' + str(arg[1]) + condi + ' is : row ' + '*' + str(
                ans.index[0] + 1), 'the row with max value in column ' + str(arg[1]) + condi

        if ops == 'argmin':
            cur_idxs = arg[0].index
            table_idxs = []
            for item in cur_idxs:
                table_idxs.append('row ' + str(item + 1))

            condi = temp_arg1 if temp_arg1 == '' else ' in ' + temp_arg1
            return 'the row with max value in column ' + '*' + str(arg[1]) + condi + ' is : row ' + '*' + str(
                ans.index[0] + 1), 'the row with max value in column ' + str(arg[1]) + condi

        if ops == 'str_hop':
            cur_idxs = arg[0].index
            table_idxs = []
            for item in cur_idxs:
                table_idxs.append('row ' + str(item + 1))
            condi = temp_arg1 if temp_arg1 == '' else ' in ' + temp_arg1
            return 'the value of column ' + '*' + str(arg[1]) + condi + ' is ' + '*' + str(
                ans), 'the value of column ' + str(arg[1]) + condi

        if ops == 'num_hop':
            cur_idxs = arg[0].index
            table_idxs = []
            for item in cur_idxs:
                table_idxs.append('row ' + str(item + 1))
            condi = temp_arg1 if temp_arg1 == '' else ' in ' + temp_arg1
            return 'the value of column ' + '*' + str(arg[1]) + condi + ' is ' + '*' + str(
                ans), 'the value of column ' + str(arg[1]) + condi

        if ops == 'most_freq':
            cur_idxs = arg[0].index
            table_idxs = []
            for item in cur_idxs:
                table_idxs.append('row ' + str(item + 1))
            condi = temp_arg1 if temp_arg1 == '' else ' in ' + temp_arg1
            return 'the most frequent value of column ' + '*' + str(arg[1]) + condi + ' is : ' + '*' + str(
                ans), 'the most frequent value of column ' + str(arg[1]) + condi

        if ops == 'half':
            cur_idxs = arg[0].index
            table_idxs = []
            for item in cur_idxs:
                table_idxs.append('row ' + str(item + 1))
            condi = temp_arg1 if temp_arg1 == '' else ' in ' + temp_arg1
            return 'the half of' + '*' + condi + ' is : ' + '*' + str(ans), 'the half of ' + condi

        if ops == 'one_third':
            cur_idxs = arg[0].index
            table_idxs = []
            for item in cur_idxs:
                table_idxs.append('row ' + str(item + 1))
            condi = temp_arg1 if temp_arg1 == '' else ' in ' + temp_arg1
            return 'one third of ' + '*' + condi + ' is : ' + '*' + str(ans), 'one third of ' + condi

        if ops == 'diff':
            if temp_arg1 != '' and temp_arg2 != '':
                return '*' + temp_arg1 + ' minus ' + '*' + temp_arg2 + ' is ' + '*' + str(
                    ans), temp_arg1 + ' minus ' + temp_arg2
            elif temp_arg1 != '' and temp_arg2 == '':
                return '*' + temp_arg1 + ' minus ' + '*' + str(arg[1]) + ' is ' + '*' + str(
                    ans), temp_arg1 + ' minus ' + str(arg[1])
            elif temp_arg1 == '' and temp_arg2 != '':
                return '*' + str(arg[0]) + ' minus ' + '*' + temp_arg2 + ' is ' + '*' + str(ans), str(
                    arg[0]) + ' minus ' + temp_arg2
            else:
                return '*' + str(arg[0]) + ' minus ' + '*' + str(arg[1]) + ' is ' + '*' + str(ans), str(
                    arg[0]) + ' minus ' + str(arg[1])
        if ops == 'add':
            if temp_arg1 != '' and temp_arg2 != '':
                return '*' + temp_arg1 + ' add ' + '*' + temp_arg2 + ' is ' + '*' + str(
                    ans), temp_arg1 + ' add ' + temp_arg2
            elif temp_arg1 != '' and temp_arg2 == '':
                return '*' + temp_arg1 + ' add ' + '*' + str(arg[1]) + ' is ' + '*' + str(
                    ans), temp_arg1 + ' add ' + str(arg[1])
            elif temp_arg1 == '' and temp_arg2 != '':
                return '*' + str(arg[0]) + ' add ' + '*' + temp_arg2 + ' is ' + '*' + str(ans), str(
                    arg[0]) + ' add ' + temp_arg2
            else:
                return '*' + str(arg[0]) + ' add ' + '*' + str(arg[1]) + ' is ' + '*' + str(ans), str(
                    arg[0]) + ' add ' + str(arg[1])

        if ops == 'greater':
            if ans:
                return '*' + str(arg[0]) + ' is larger than ' + '*' + str(arg[1]), str(
                    arg[0]) + ' is larger than ' + str(arg[1])
            else:
                return '*' + str(arg[0]) + ' is less than ' + '*' + str(arg[1]), str(arg[0]) + ' is less than ' + str(
                    arg[1])
        if ops == 'less':
            if ans:
                return '*' + str(arg[0]) + ' is less than ' + '*' + str(arg[1]), str(arg[0]) + ' is less than ' + str(
                    arg[1])
            else:
                return '*' + str(arg[0]) + ' is larger than ' + '*' + str(arg[1]), str(
                    arg[0]) + ' is larger than ' + str(arg[1])
        if ops == 'and':
            if ans:
                return '*' + temp_arg1 + ' and ' + '*' + temp_arg2 + ' exist', temp_arg1 + ' and ' + temp_arg2 + ' exist'
            else:
                return '*' + temp_arg1 + ' and ' + '*' + temp_arg2 + ' not exist', temp_arg1 + ' and ' + temp_arg2 + ' not exist'
        if ops == 'filter_str_eq' or ops == 'filter_eq':
            cur_idxs = arg[0].index
            table_idxs = []
            for item in cur_idxs:
                table_idxs.append('row ' + str(item + 1))

            if len(ans) > 0:
                ret = []
                for item in ans.index:
                    ret.append('row ' + str(item + 1))

                x = 'the table' if temp_arg1 == '' else temp_arg1
                return '*' + x + ' where column ' + '*' + str(arg[1]) + ' with value ' + '*' + str(
                    arg[2]) + ' are ' + ' , '.join(ret), x + ' where column ' + str(arg[1]) + ' with value ' + str(
                    arg[2])
            else:
                x = 'the table' if temp_arg1 == '' else temp_arg1
                return '*' + x + ' where column ' + '*' + str(arg[1]) + ' with value ' + '*' + str(
                    arg[2]) + ' are empty', x + ' where column ' + str(arg[1]) + ' with value ' + str(arg[2])

        if ops == 'filter_str_not_eq' or ops == 'filter_not_eq':
            cur_idxs = arg[0].index
            table_idxs = []
            for item in cur_idxs:
                table_idxs.append('row ' + str(item + 1))
            if len(ans) > 0:
                ret = []
                for item in ans.index:
                    ret.append('row ' + str(item + 1))

                x = 'the table' if temp_arg1 == '' else temp_arg1
                return '*' + x + ' where column ' + '*' + str(arg[1]) + ' not with value ' + '*' + str(
                    arg[2]) + ' are ' + ' , '.join(ret), x + ' where column ' + str(arg[1]) + ' not with value ' + str(
                    arg[2])
            else:
                x = 'the table' if temp_arg1 == '' else temp_arg1
                return '*' + x + ' where column ' + '*' + str(arg[1]) + ' not with value ' + '*' + str(
                    arg[2]) + ' are empty', x + ' where column ' + str(arg[1]) + ' not with value ' + str(arg[2])

        if ops == 'filter_less':
            cur_idxs = arg[0].index
            table_idxs = []
            for item in cur_idxs:
                table_idxs.append('row ' + str(item + 1))
            if len(ans) > 0:
                ret = []
                for item in ans.index:
                    ret.append('row ' + str(item + 1))

                x = 'the table' if temp_arg1 == '' else temp_arg1
                return '*' + x + ' where column ' + '*' + str(arg[1]) + ' less than ' + '*' + str(
                    arg[2]) + ' are ' + ' , '.join(ret), x + ' where column ' + str(arg[1]) + ' less than ' + str(
                    arg[2])
            else:
                x = 'the table' if temp_arg1 == '' else temp_arg1
                return '*' + x + ' where column ' + '*' + str(arg[1]) + ' less than ' + '*' + str(
                    arg[2]) + ' are empty', x + ' where column ' + str(arg[1]) + ' less than ' + str(arg[2])

        if ops == 'filter_greater':
            cur_idxs = arg[0].index
            table_idxs = []
            for item in cur_idxs:
                table_idxs.append('row ' + str(item + 1))

            if len(ans) > 0:
                ret = []
                for item in ans.index:
                    ret.append('row ' + str(item + 1))

                x = 'the table' if temp_arg1 == '' else temp_arg1
                return '*' + x + ' where column ' + '*' + str(arg[1]) + ' greater than ' + '*' + str(
                    arg[2]) + ' are ' + ' , '.join(ret), x + ' where column ' + str(arg[1]) + ' greater than ' + str(
                    arg[2])
            else:
                x = 'the table' if temp_arg1 == '' else temp_arg1
                return '*' + x + ' where column ' + '*' + str(arg[1]) + ' greater than ' + '*' + str(
                    arg[2]) + ' are empty', x + ' where column ' + str(arg[1]) + ' greater than ' + str(arg[2])

        if ops == 'all_str_eq' or ops == 'all_eq':
            if ans:
                return 'values in column ' + '*' + str(arg[1]) + ' are equal to ' + '*' + str(
                    arg[2]), 'values in column ' + str(arg[1]) + ' are equal to ' + str(arg[2])
            else:
                return 'Not all values in column ' + '*' + str(arg[1]) + ' are equal to ' + '*' + str(
                    arg[2]), 'Not all values in column ' + str(arg[1]) + ' are equal to ' + str(arg[2])

        if ops == 'all_str_not_eq' or ops == 'all_not_eq':
            if ans:
                return 'Not all values in column ' + '*' + str(arg[1]) + ' are equal to ' + '*' + str(
                    arg[2]), 'Not all values in column ' + str(arg[1]) + ' are equal to ' + str(arg[2])
            else:
                return 'values in column ' + '*' + str(arg[1]) + ' are equal to ' + '*' + str(
                    arg[2]), 'values in column ' + str(arg[1]) + ' are equal to ' + str(arg[2])
        if ops == 'all_greater':
            if ans:
                return 'values in column ' + '*' + str(arg[1]) + ' are larger than ' + '*' + str(
                    arg[2]), 'values in column ' + str(arg[1]) + ' are larger than ' + str(arg[2])
            else:
                return 'Not all values in column ' + '*' + str(arg[1]) + ' are larger than ' + '*' + str(
                    arg[2]), 'Not all values in column ' + str(arg[1]) + ' are larger than ' + str(arg[2])
        if ops == 'all_less':
            if ans:
                return 'values in column ' + '*' + str(arg[1]) + ' are less than ' + '*' + str(
                    arg[2]), 'values in column ' + str(arg[1]) + ' are less than ' + str(arg[2])
            else:
                return 'Not all values in column ' + '*' + str(arg[1]) + ' are less than ' + '*' + str(
                    arg[2]), 'Not all values in column ' + str(arg[1]) + ' are less than ' + str(arg[2])
        if ops == 'all_greater_eq':
            if ans:
                return 'values in column ' + '*' + str(arg[1]) + ' are larger than or equal to ' + '*' + str(
                    arg[2]), 'values in column ' + str(arg[1]) + ' are larger than or equal to ' + str(arg[2])
            else:
                return 'Not all values in column ' + '*' + str(arg[1]) + ' are larger than or equal to ' + '*' + str(
                    arg[2]), 'Not all values in column ' + str(arg[1]) + ' are larger than or equal to ' + str(arg[2])
        if ops == 'all_less_eq':
            if ans:
                return 'values in column ' + '*' + str(arg[1]) + ' are less than or equal to ' + '*' + str(
                    arg[2]), 'values in column ' + str(arg[1]) + ' are less than or equal to ' + str(arg[2])
            else:
                return 'Not all values in column ' + '*' + str(arg[1]) + ' are less than or equal to ' + '*' + str(
                    arg[2]), 'Not all values in column ' + str(arg[1]) + ' are less than or equal to ' + str(arg[2])
        return "", ""

    def opArgExec(self, ar, child_args):
        cur_op = ar.val
        cur_arg_num = 0
        if ar.val in ['within', 'not_within', 'filter_eq', 'filter_not_eq', 'filter_less',
                      'filter_greater', 'filter_greater_eq', 'filter_less_eq', 'all_eq', 'all_not_eq',
                      'all_less', 'all_less_eq', 'all_greater', 'all_greater_eq']:
            if type(child_args[1]) != str:
                child_args[1] = str(child_args[1])
        if ar.val in ['uniq', 'avg', 'sum', 'max', 'min', 'argmax', 'argmin',
                      'hop', 'most_freq', 'num_hop']:
            if len(child_args) > 1 and type(child_args[1]) != str:
                if type(child_args[1]) == float:
                    child_args[1] = '{:.2f}'.format(child_args[1])
                else:
                    child_args[1] = str(child_args[1])

        if ar.val == 'eq':
            cur_arg_num = 2
            arg0 = child_args[0]
            try:
                arg1 = child_args[1]
            except:
                return None, None, None
            if type(arg0) == str:
                if type(arg1) == np.float64 or type(arg1) == np.int64:
                    cur_op = 'eq'
                    child_args[0] = float(arg0)
                    if type(arg1) == np.int64:
                        child_args[0] = int(child_args[0])
                else:
                    cur_op = 'str_eq'
            else:
                if type(arg1) == str:
                    if arg1.isdigit():
                        child_args[1] = int(child_args[1])
                        cur_op = 'eq'
                    else:
                        cur_op = 'str_eq'
                        child_args[0] = str(child_args[0])
                else:
                    cur_op = 'eq'
        if ar.val == 'not_eq':
            cur_arg_num = 2
            arg0 = child_args[0]
            if type(arg0) == str:
                cur_op = 'not_str_eq'
            else:
                cur_op = 'not_eq'
        if ar.val == 'count':
            cur_arg_num = 1
            cur_op = ar.val
        if ar.val == 'within':
            cur_arg_num = 3
            arg0 = child_args[-1]
            if type(arg0) == str:
                cur_op = 'within_s_s'
            else:
                cur_op = 'within_n_n'
        if ar.val == 'not_within':
            cur_arg_num = 3
            arg0 = child_args[-1]
            if type(arg0) == str:
                cur_op = 'not_within_s_s'
            else:
                cur_op = 'not_within_n_n'
        if ar.val == 'none' or ar.val == 'half' or ar.val == 'one_third':
            cur_arg_num = 1
        if ar.val == 'only' or ar.val == 'several' or ar.val == 'zero' or ar.val == 'top' or ar.val == 'bottom':
            cur_arg_num = 1
        if ar.val == 'after' or ar.val == 'before':
            cur_arg_num = 3
        if ar.val == 'first' or ar.val == 'second' or ar.val == 'most_freq' or \
                        ar.val == 'third' or ar.val == 'fourth' or ar.val == 'fifth' or ar.val == 'last':
            cur_arg_num = 2
        if ar.val == 'uniq':
            cur_arg_num = 2
            arg0 = child_args[1]
            if type(arg0) == str:
                cur_op = 'uniq_str'
            else:
                cur_op = 'uniq_num'
        if ar.val == 'hop':
            cur_arg_num = 2
            try:
                arg0 = child_args[1]
            except:
                return None, None, None
            if type(arg0) == str:
                cur_op = 'str_hop'
            else:
                cur_op = 'num_hop'
        if ar.val == 'avg' or ar.val == 'sum' or ar.val == 'max' or ar.val == 'min' or ar.val == 'argmax' or ar.val == 'argmin':
            cur_arg_num = 2
        if ar.val == 'diff' or ar.val == 'add' or ar.val == 'greater' or ar.val == 'less' or ar.val == 'and':
            cur_arg_num = 2
        if ar.val == 'filter_eq':
            cur_arg_num = 3
            arg0 = child_args[2]
            try:
                arg2 = child_args[0][child_args[1]].dtype
            except:
                return None, None, None

            if type(arg0) == str:
                cur_op = 'filter_str_eq'
            else:
                if arg2 != float and arg2 != int:
                    child_args[2] = str(arg0)
                    cur_op = 'filter_str_eq'
                else:
                    cur_op = 'filter_eq'
        if ar.val == 'filter_not_eq':
            cur_arg_num = 3
            arg0 = child_args[2]
            arg2 = child_args[0][child_args[1]].dtype
            if type(arg0) == str:
                cur_op = 'filter_str_not_eq'
            else:
                if arg2 != float and arg2 != int:
                    child_args[2] = str(arg0)
                    cur_op = 'filter_str_not_eq'
                else:
                    cur_op = 'filter_not_eq'
        if ar.val == 'all_eq':
            cur_arg_num = 3
            arg0 = child_args[2]
            arg2 = child_args[0][child_args[1]].dtype

            if type(arg0) == str:
                cur_op = 'all_str_eq'
            else:
                if arg2 != float and arg2 != int:
                    child_args[2] = str(arg0)
                    cur_op = 'all_str_eq'
                else:
                    cur_op = 'all_eq'
        if ar.val == 'all_not_eq':
            cur_arg_num = 3
            arg0 = child_args[2]
            arg2 = child_args[0][child_args[1]].dtype

            if type(arg0) == str:
                if type(arg0) != str:
                    child_args[2] = str(child_args[2])
                cur_op = 'all_str_not_eq'
            else:
                if arg2 != float and arg2 != int:
                    if type(arg0) != str:
                        child_args[2] = str(child_args[2])
                    cur_op = 'all_str_not_eq'
                else:
                    cur_op = 'all_not_eq'

        if ar.val == 'filter_less' or ar.val == 'filter_greater' or ar.val == 'filter_greater_eq' or ar.val == 'filter_less_eq' \
                or ar.val == 'all_less' or ar.val == 'all_less_eq' or ar.val == 'all_greater' or ar.val == 'all_greater_eq':
            cur_arg_num = 3

        cur_ans = self.execute(cur_op, child_args, cur_arg_num)
        return cur_ans, cur_op, cur_arg_num

    def convertArg0330(self, ops, args, table, cols, mapping, temps):
        ret = []
        ret_temps = []
        ret_temp_conditions = []

        def check_digit(num_str):
            if (num_str[0] == '១' or num_str[0] == '០' or num_str[0] == '២'
                or num_str[0] == '៣' or num_str[0] == '៤' or num_str[0] == '៥'):
                return False
            match = re.search(r'\d+', num_str)
            if match:
                span = match.span()
                if (span[1] - span[0]) == len(num_str):
                    return True
            match = re.search(r'\d+\.\d+', num_str)
            if match:
                span = match.span()
                if (span[1] - span[0]) == len(num_str):
                    return True
            match = re.search(r'\d+\.\d+e\+\d+', num_str)
            if match:
                span = match.span()
                if (span[1] - span[0]) == len(num_str):
                    return True
            return False

        for ar in args:
            if len(ar.args) == 0:
                if ar.val == 'all_rows':
                    ret.append(table)
                elif check_digit(ar.val):
                    if '.' in ar.val:
                        ret.append(float(ar.val))
                    else:
                        ret.append(int(ar.val))
                else:
                    ret.append(ar.val)
                ret_temps.append(ar.temp)
                ret_temp_conditions.append(ar.temp)

            else:
                child_ops = ar.val

                child_args, child_temps, child_conditions = self.convertArg0330(child_ops, ar.args, table, cols,
                                                                                mapping, temps)
                # execute
                cur_ans, cur_op, cur_arg_num = self.opArgExec(ar, child_args)  # ar: op-node, child_args: variables

                # get the evidence sentence
                if cur_ans is not None:
                    cur_exec_temp = ""
                    if len(child_args) == 3:
                        cur_exec_temp, cur_exec_conditions = self.convertAns2Template(cur_op, child_args, cur_ans,
                                                                                      child_conditions[0],
                                                                                      child_conditions[1],
                                                                                      child_conditions[2], cols)
                    if len(child_args) == 2:
                        cur_exec_temp, cur_exec_conditions = self.convertAns2Template(cur_op, child_args, cur_ans,
                                                                                      child_conditions[0],
                                                                                      child_conditions[1], "", cols)
                    if len(child_args) == 1:
                        cur_exec_temp, cur_exec_conditions = self.convertAns2Template(cur_op, child_args, cur_ans,
                                                                                      child_conditions[0],
                                                                                      "", "", cols)
                    ar.temp = cur_exec_temp
                    ret.append(cur_ans)
                    ret_temps.append(cur_exec_temp)
                    ret_temp_conditions.append(cur_exec_conditions)

                    temps.append(cur_exec_temp)  # save the evidence sentence

        return ret, ret_temps, ret_temp_conditions

    def convertArg(self, ops, args, table, cols, mapping, temps):
        ret = []
        ret_temps = []

        def check_digit(num_str):
            if (num_str[0] == '១' or num_str[0] == '០' or num_str[0] == '២'
                or num_str[0] == '៣' or num_str[0] == '៤' or num_str[0] == '៥'):
                return False
            match = re.search(r'\d+', num_str)
            if match:
                span = match.span()
                if (span[1] - span[0]) == len(num_str):
                    return True
            match = re.search(r'\d+\.\d+', num_str)
            if match:
                span = match.span()
                if (span[1] - span[0]) == len(num_str):
                    return True
            match = re.search(r'\d+\.\d+e\+\d+', num_str)
            if match:
                span = match.span()
                if (span[1] - span[0]) == len(num_str):
                    return True

            return False

        for ar in args:
            if len(ar.args) == 0:
                if ar.val == 'all_rows':
                    ret.append(table)
                elif check_digit(ar.val):
                    if '.' in ar.val:
                        ret.append(float(ar.val))
                    else:
                        ret.append(int(ar.val))
                else:
                    ret.append(ar.val)
                ret_temps.append(ar.temp)
            else:
                child_ops = ar.val
                child_args, child_temps = self.convertArg(child_ops, ar.args, table, cols, mapping, temps)
                cur_ans, cur_op, cur_arg_num = self.opArgExec(ar, child_args)
                if cur_ans is not None:
                    cur_exec_temp = ""
                    if len(child_args) == 3:
                        cur_exec_temp = self.convertAns2Template(cur_op, child_args, cur_ans, child_temps[0],
                                                                 child_temps[1], child_temps[2], cols)
                    if len(child_args) == 2:
                        cur_exec_temp = self.convertAns2Template(cur_op, child_args, cur_ans, child_temps[0],
                                                                 child_temps[1], "", cols)
                    if len(child_args) == 1:
                        cur_exec_temp = self.convertAns2Template(cur_op, child_args, cur_ans, child_temps[0],
                                                                 "", "", cols)
                    ar.temp = cur_exec_temp
                    ret.append(cur_ans)
                    ret_temps.append(cur_exec_temp)
                    temps.append(cur_exec_temp)
        return ret, ret_temps

    def traverse_tree_without_root(self, root, parent, lst, index_dict):
        '''v1: without the verb of root node, RootLR order, only the nearest child node can update the parent node, directed'''
        global index
        if parent is None:  # root
            index = 0
        elif len(root.args) != 0:  # not root
            root.id = index
            index_dict[str(root.id) + '_' + str(root.val)] = []
        if len(root.args) != 0:  # has child node
            if root.temp != "" and parent is not None:  # has verb
                lst.append(root.temp)
                index += 1
            for item in root.args:  # each child node
                if item.temp != "" and parent is not None:
                    index_dict[str(root.id) + '_' + str(root.val)].append(index)
                self.traverse_tree_without_root(item, root, lst, index_dict)

    def traverse_tree_with_root(self, root, parent, lst, index_dict):
        '''v2: with the verb of root node, RootLR order, undirected'''
        global index
        if parent is None:  # root
            index = 0
        if len(root.args) != 0:  # has child node
            root.id = index
            index_dict[str(root.id) + '_' + str(root.val)] = []
            if root.temp != "":  # has verb
                lst.append(root.temp)
                index += 1
            for item in root.args:  # each child node
                if item.temp != "":
                    index_dict[str(root.id) + '_' + str(root.val)].append(index)
                self.traverse_tree_with_root(item, root, lst, index_dict)

    def traverseTreeTemplates(self, root, table_name):
        if root is None:
            return None, [], None
        if len(root.args) == 0:
            return None, [], None

        t = pandas.read_csv('../../data/all_csv/{}'.format(table_name), delimiter="#")
        cols = t.columns
        mapping = {i: "num" if isnumber(t) else "str" for i, t in enumerate(t.dtypes)}

        total_temps = []  # save evidence sentences
        child_args, child_temps, child_temps_conditions = self.convertArg0330(root, root.args, t, cols, mapping,
                                                                              total_temps)

        if add_root_verb:
            cur_ans, cur_op, cur_arg_num = self.opArgExec(root, child_args)
        if add_root_verb and cur_ans is not None:
            cur_exec_temp = ""
            if len(child_args) == 3:
                cur_exec_temp, _ = self.convertAns2Template(cur_op, child_args, cur_ans, child_temps[0], child_temps[1],
                                                            child_temps[2], cols)
            if len(child_args) == 2:
                cur_exec_temp, _ = self.convertAns2Template(cur_op, child_args, cur_ans, child_temps[0], child_temps[1],
                                                            "", cols)
            if len(child_args) == 1:
                cur_exec_temp, _ = self.convertAns2Template(cur_op, child_args, cur_ans, child_temps[0], "", "", cols)
            root.temp = cur_exec_temp

        new_order_total_temps = []
        index_dict = {}
        if add_root_verb:
            self.traverse_tree_with_root(root, None, new_order_total_temps, index_dict)
        else:
            self.traverse_tree_without_root(root, None, new_order_total_temps, index_dict)

        final_index_dict = []
        for i in range(len(index_dict.items())):
            final_index_dict.append([])
        for k, v in index_dict.items():
            k = int(k.split('_')[0])
            if len(new_order_total_temps) != 0:
                final_index_dict[k] = v

        return cur_ans, new_order_total_temps, root, final_index_dict

    def findIndex(self, expression, st, ed0):
        stack = []
        ed = -1
        for i in range(st, ed0):
            if expression[i] == '{':
                stack.append(i)
            else:
                if expression[i] == '}':
                    if stack[-1] < i:
                        stack.pop(-1)
                        if len(stack) == 0:
                            ed = i
                            return ed
        return ed


def preprocessRawData(fname, fmatch, fsave, attn_save, train=False, thresh=0.0):
    '''
    :param fname:  selected prog
    :param fmatch: original training data
    :param fsave:  save file to
    :param train:  False
    :param thresh: threshold
    :return: 
    '''

    ret = []
    ret_dict = dict()
    has_program = 0
    cur_express = ExpressionTree()
    over_thresh = 0
    max_ = 0

    max_lst = 0
    max_set = 0
    max_sent_len = 0
    avg_lst = 0
    avg_set = 0
    avg_sent_len = 0
    de_duplicate_count = 0

    with open(fname, 'r', encoding='utf-8') as fin:
        raw_inputs = json.load(fin)

        hgcn_attn_lst_verb_verb = {}
        hgcn_attn_lst_verb_entity = {}
        hgcn_attn_lst_cls_verb = {}

        assert max_entity_num % max_verb_num == 0
        avg_entity = int(max_entity_num / max_verb_num)

        count_meaningful_matrix = 0

        for i, raw_dict in enumerate(tqdm(raw_inputs)):

            raw_data = raw_dict['originaldata'].split('\t')

            input_tsv = raw_data[0].strip()
            input_true_label = True if raw_data[2] == '1' else False
            input_program = raw_data[5] if raw_data[5] != 'no program' else ''
            input_sent_linked = raw_data[4].strip()
            processed_sent = parse_fact(input_sent_linked)
            cur_key = input_tsv + processed_sent

            tmp_attn_matrix_verb_verb = torch.eye(matrix_len)
            tmp_attn_matrix_verb_entity = torch.eye(matrix_len)
            tmp_attn_matrix_cls_verb = torch.eye(matrix_len)

            sim = float(raw_dict['similarity'])
            if sim >= thresh:
                over_thresh += 1

                if input_program == '':
                    if input_tsv + processed_sent in ret_dict:
                        hgcn_attn_lst_verb_verb[cur_key] = tmp_attn_matrix_verb_verb
                        hgcn_attn_lst_verb_entity[cur_key] = tmp_attn_matrix_verb_entity
                        hgcn_attn_lst_cls_verb[cur_key] = tmp_attn_matrix_cls_verb
                        continue
                    ret_dict[input_tsv + processed_sent] = ''
                    hgcn_attn_lst_verb_verb[cur_key] = tmp_attn_matrix_verb_verb
                    hgcn_attn_lst_verb_entity[cur_key] = tmp_attn_matrix_verb_entity
                    hgcn_attn_lst_cls_verb[cur_key] = tmp_attn_matrix_cls_verb
                    continue
                dealt_input_program = []
                word = ""
                cnt = 0
                jj = 0

                # -- to get dealt_input_program --
                while jj < len(input_program):
                    if input_program[jj] != ' ':
                        word += input_program[jj]
                    else:
                        if word != '':
                            if jj > 0:
                                if input_program[jj - 1] == '{' or input_program[jj - 1] == '}':
                                    dealt_input_program.append(word)
                                    word = ''
                                    jj += 1
                                    continue
                            while jj < len(input_program) - 1 and input_program[jj + 1] == ' ':
                                word += ' '
                                jj += 1
                            if word == '; ':
                                dealt_input_program.append(';')
                                word = ' '
                            else:
                                dealt_input_program.append(word)
                                word = ''
                        else:
                            word = ''

                    jj += 1

                # -- build tree --
                node = cur_express.buildTree(dealt_input_program, 0, len(
                    dealt_input_program))  # dealt_input_prog: ['and', '{', 'only', '{', 'filter_eq', '{', 'all_rows', ';', 'mission', ';',.......
                # -- get verb   --
                node_ans, node_temps_ori, node_root, index_dict = cur_express.traverseTreeTemplates(node, input_tsv)

                node_temps = []
                for verb in node_temps_ori:
                    verb_new = ''
                    for idx, v in enumerate(verb):
                        if idx + 1 < len(verb):
                            if verb[idx + 1] != '*' or v == ' ':
                                verb_new += v
                        else:
                            verb_new += v
                    node_temps.append(verb_new)

                '''node_temps is the verbalized sentence list, 
                index_dict is like [[1], [], [3], [4], []] to build attention mask'''

                # --- verb & verb ---
                for kk in range(len(index_dict)):
                    for item in index_dict[kk]:
                        tmp_attn_matrix_verb_verb[kk + 1][item + 1] = 1
                        if matrix_symmetric:
                            tmp_attn_matrix_verb_verb[item + 1][kk + 1] = 1
                count_meaningful_matrix += 1

                # --- cls & verb ---
                for ii in range(len(node_temps)):
                    tmp_attn_matrix_cls_verb[0][ii + 1] = 1

                # --- verb & entity ---
                for ii in range(len(node_temps)):
                    e_1 = max_verb_num + 1 + avg_entity * ii
                    for jj in range(avg_entity):
                        tmp_attn_matrix_verb_entity[ii + 1][e_1 + jj] = 1

                tmp_lst = set(node_temps)
                avg_lst += len(node_temps)
                avg_set += len(tmp_lst)
                if len(tmp_lst) < len(node_temps):
                    de_duplicate_count += 1
                if len(node_temps) > max_lst:
                    max_lst = len(node_temps)
                if len(tmp_lst) > max_set:
                    max_set = len(tmp_lst)
                len_ = 0
                for sent in node_temps:
                    avg_sent_len += len(sent.split(' '))
                    len_ += len(sent.split(' '))
                if len_ > max_sent_len:
                    max_sent_len = len_

                if node_ans is not None:
                    if train:
                        if node_ans == input_true_label:
                            ret_dict[input_tsv + processed_sent] = '\t' + str(len(node_temps)) + '\t' + '\t'.join(
                                node_temps)
                            has_program += 1
                        else:
                            ret_dict[input_tsv + processed_sent] = ''
                    else:
                        ret_dict[input_tsv + processed_sent] = '\t' + str(len(node_temps)) + '\t' + '\t'.join(
                            node_temps)
                        has_program += 1
                else:
                    ret_dict[input_tsv + processed_sent] = ''
            else:
                ret_dict[input_tsv + processed_sent] = ''

            hgcn_attn_lst_verb_verb[cur_key] = tmp_attn_matrix_verb_verb
            hgcn_attn_lst_verb_entity[cur_key] = tmp_attn_matrix_verb_entity
            hgcn_attn_lst_cls_verb[cur_key] = tmp_attn_matrix_cls_verb
        fin.close()
    lines = []
    match_num = 0

    # statistics
    all_count = 0
    add_count = 0
    original_len = 0
    add_len = 0
    not_match = 0
    short = 0

    reorder_hgcn_attn_lst_verb_verb = []
    reorder_hgcn_attn_lst_verb_entity = []
    reorder_hgcn_attn_lst_cls_verb = []

    with open(fmatch, 'r', encoding='utf-8') as fin:
        for line in fin:
            all_count += 1
            words = line.strip().split('\t')
            cur_key = words[0] + words[-2]
            original_len += len(words[-3].split(' '))
            if cur_key in ret_dict:
                items = ret_dict[cur_key].split(' ')
                if len(items) > 1:
                    add_count += 1
                    add_len += len(items)
                else:
                    short += 1
                words[-3] = words[-3] + ret_dict[cur_key]
                match_num += 1
                reorder_hgcn_attn_lst_verb_verb.append(hgcn_attn_lst_verb_verb[cur_key])
                reorder_hgcn_attn_lst_verb_entity.append(hgcn_attn_lst_verb_entity[cur_key])
                reorder_hgcn_attn_lst_cls_verb.append(hgcn_attn_lst_cls_verb[cur_key])
            else:
                not_match += 1
                if cur_key in hgcn_attn_lst_verb_verb:
                    reorder_hgcn_attn_lst_verb_verb.append(hgcn_attn_lst_verb_verb[cur_key])
                    reorder_hgcn_attn_lst_verb_entity.append(hgcn_attn_lst_verb_entity[cur_key])
                    reorder_hgcn_attn_lst_cls_verb.append(hgcn_attn_lst_cls_verb[cur_key])
                else:
                    reorder_hgcn_attn_lst_verb_verb.append(torch.eye(matrix_len))
                    reorder_hgcn_attn_lst_verb_entity.append(torch.eye(matrix_len))
                    reorder_hgcn_attn_lst_cls_verb.append(torch.eye(matrix_len))

            lines.append('\t'.join(words) + '\n')
        fin.close()

    # save tensor
    # -> save verb & verb
    attn_save_verb_verb = os.path.join(output_dir, attn_save + '_verb_verb.pth')
    torch.save(reorder_hgcn_attn_lst_verb_verb, attn_save_verb_verb)
    # -> save verb & entity
    attn_save_verb_entity = os.path.join(output_dir, attn_save + '_verb_entity.pth')
    torch.save(reorder_hgcn_attn_lst_verb_entity, attn_save_verb_entity)
    # -> save cls & verb
    attn_save_cls_verb = os.path.join(output_dir, attn_save + '_cls_verb.pth')
    torch.save(reorder_hgcn_attn_lst_cls_verb, attn_save_cls_verb)

    with open(fsave, 'w', encoding='utf-8') as fout:
        for line in lines:
            fout.write(line)
        fout.close()

    print(
        '>> lines#: %d, add#: %d, percent: %.4f, thresh: %.3f' % (all_count, add_count, add_count / all_count, thresh))


# -------- entry ---------
matrix_symmetric = True  # True: undirected  False: directed
add_root_verb = True  # (root) verb
max_verb_num = 10
if add_root_verb:
    max_verb_num += 1

max_entity_num = 22  # entity (avg=2)
matrix_len = max_verb_num + max_entity_num

add_ques_in_matrix = True
max_ques_num = 0
if add_ques_in_matrix:
    max_ques_num += 1
    matrix_len += 1

# to create attention matrix for hgcn:
# (1) matrix-1: verb & entity
# (2) matrix-2: verb & verb
# (3) matrix-3: question (cls from table-fact-bert) & verb
# we can get the size of attention matrix based on the above params:
print("question: {}, verb: {}, entity: {} == {}".format(max_ques_num, max_verb_num, max_entity_num, matrix_len))

parser = argparse.ArgumentParser()
parser.add_argument("--selected_prog_dir", help='the selected programs (output of program selection)')
parser.add_argument("--table_bert_dir", help='the input of table bert')
parser.add_argument("--save_root_dir", help='the results of verb')
args = parser.parse_args()

selected_prog_dir = args.selected_prog_dir
table_bert_dir = args.table_bert_dir
save_root_dir = args.save_root_dir


dev_path = os.path.join(selected_prog_dir, 'dev_program_select_only_true.json')
train_path = os.path.join(selected_prog_dir, 'train_program_select_only_true.json')
test_path = os.path.join(selected_prog_dir, 'std_test_program_select_only_true.json')
simple_test_path = os.path.join(selected_prog_dir, 'simple_test_program_select_only_true.json')
complex_test_path = os.path.join(selected_prog_dir, 'complex_test_program_select_only_true.json')
small_test_path = os.path.join(selected_prog_dir, 'small_test_program_select_only_true.json')

m_str = 'undirected' if matrix_symmetric else 'directed'
output_dir = os.path.join(save_root_dir, 'tsv_data_verb')
if not os.path.exists(save_root_dir):
    os.mkdir(save_root_dir)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

preprocessRawData(train_path,
                  os.path.join(table_bert_dir, 'train.tsv'),
                  os.path.join(output_dir, 'train_margin.tsv'),
                  'train_margin_attn_matrix',
                  thresh=0.25)
print(">> finish train\n")

preprocessRawData(dev_path,
                  os.path.join(table_bert_dir, 'dev.tsv'),
                  os.path.join(output_dir, 'dev_margin.tsv'),
                  'dev_margin_attn_matrix',
                  thresh=0.16)
print(">> finish dev\n")

preprocessRawData(test_path,
                  os.path.join(table_bert_dir, 'test.tsv'),
                  os.path.join(output_dir, 'test_margin.tsv'),
                  'test_margin_attn_matrix',
                  thresh=0.19)
print(">> finish test\n")

preprocessRawData(simple_test_path,
                  os.path.join(table_bert_dir, 'simple_test.tsv'),
                  os.path.join(output_dir, 'simple_test_margin.tsv'),
                  'simple_test_margin_attn_matrix',
                  thresh=0.19)
print(">> finish simple_test\n")

preprocessRawData(complex_test_path,
                  os.path.join(table_bert_dir, 'complex_test.tsv'),
                  os.path.join(output_dir, 'complex_test_margin.tsv'),
                  'complex_test_margin_attn_matrix',
                  thresh=0.19)
print(">> finish complex_test\n")

preprocessRawData(small_test_path,
                  os.path.join(table_bert_dir, 'small_test.tsv'),
                  os.path.join(output_dir, 'small_test_margin.tsv'),
                  'small_test_margin_attn_matrix',
                  thresh=0.25)
print(">> finish small_test\n")
