import json
import os
import random
import re

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, ErnieModel
from tqdm import tqdm

from utils.logger import setup_logger

# tokenizer = BertTokenizer.from_pretrained(r"D:\work\asrapp\puncModel\punctuationSelf\ernie-3.0-nano-zh")
# model = ErnieModel.from_pretrained(r"D:\work\asrapp\puncModel\punctuationSelf\ernie-3.0-nano-zh")


logger = setup_logger(__name__)

__all__ = ["PuncDatasetFromErnieTokenizer", "collate_fn"]


class PuncDatasetFromErnieTokenizer(Dataset):
    def __init__(self, data_path, punc_path, pretrained_token='ernie-3.0-nano-zh', max_seq_len=100):
        super().__init__()
        self.inputs_data = []
        self.labels = []
        self.cache_data_path = os.path.join(os.path.dirname(data_path), f'{os.path.basename(data_path)}.cache')
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_token)
        self.paddingID = self.tokenizer.pad_token_id
        self.max_seq_len = max_seq_len
        # 加载标点符号字典，因为开头还有空格
        # self.punc2id = self.load_vocab(punc_path, extra_word_list=[" "])
        # self.id2punc = {k: v for (v, k) in self.punc2id.items()}
        # 预处理数据
        self.preprocess(data_path)

    def __len__(self):
        return len(self.inputs_data)

    def __getitem__(self, index):
        inputs_data = np.array(self.inputs_data[index][:self.max_seq_len], dtype='int64')
        labels = np.array(self.labels[index][:self.max_seq_len], dtype='int64')
        return inputs_data, labels

    @staticmethod
    def load_vocab(vocab_path, extra_word_list=None):
        if extra_word_list is None:
            extra_word_list = []
        n = len(extra_word_list)
        with open(vocab_path, encoding='utf-8') as vf:
            vocab = {word.strip(): i + n for i, word in enumerate(vf)}
        for i, word in enumerate(extra_word_list):
            vocab[word] = i
        return vocab

    def preprocess(self, data_path: str):
        if not os.path.exists(self.cache_data_path):
            logger.info(f'{self.cache_data_path}不存在，正在重新生成，时间比较长，请耐心等待...')
            txt_seqs = open(data_path, encoding='utf-8').read().splitlines()
            split_txt_seqs = []
            pattern1 = re.compile('[……？！”“（）《》【】、——：￥$%^&@()+-=-]')
            pattern2 = re.compile("[”“（）《》【】、，——：]")
            pattern3 = re.compile(r"[A-Za-z]")
            pattern4 = re.compile(r"\s+")
            for i in range(len(txt_seqs)):
                tmp_seq = pattern1.sub(" ", txt_seqs[i], count=0)
                # tmp_seq = pattern2.sub(" ", tmp_seq, count=0)
                tmp_seq = pattern3.sub("", tmp_seq, count=0)
                tmp_seq = pattern4.sub(" ", tmp_seq, count=0)
                if "。" in tmp_seq:
                    for seq in tmp_seq.split("。"):
                        seq = seq.strip()
                        if len(seq) == 0:
                            continue
                        if "，" in seq:
                            for s in seq.split("，"):
                                s = s.strip()
                                if len(s) == 0:
                                    continue
                                split_txt_seqs.append(s)
                        else:
                            split_txt_seqs.append(seq)
                else:
                    split_txt_seqs.append(tmp_seq.strip())
            filter_txt_seqs = []
            for seq in split_txt_seqs:
                if " " in seq:
                    filter_txt_seqs.append(seq)

            # 对数据按照从短到长排序
            txt_seqs = sorted(filter_txt_seqs, key=lambda k: len(k))
            for text in tqdm(txt_seqs):
                label, input_data = [], []
                for i in range(len(text)-1):
                    if text[i] == " ": continue
                    if text[i+1] == " ":
                        label.append(1)
                    else:
                        label.append(0)
                    token = self.tokenizer(text[i])
                    input_data.append(token["input_ids"][1])
                input_data.append(self.tokenizer(text[-1])["input_ids"][1])
                label.append(1)
                self.inputs_data.append(input_data)
                self.labels.append(label)
            data = {'inputs_data': self.inputs_data, 'labels': self.labels}
            with open(self.cache_data_path, 'w', encoding='utf-8') as f:
                json.dump(data, f)
        else:
            logger.info(f'正在加载：{self.cache_data_path}')
            # 读取之前制作好的数据，如果是更换了数据集，需要删除这几个缓存文件
            with open(self.cache_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.inputs_data = data['inputs_data']
                self.labels = data['labels']

        if len(self.inputs_data) != len(self.labels):
            assert 'error: length input_data != label'


# 对一个batch的数据处理
def collate_fn(batch):
    # 找出数据长度最长的
    batch = sorted(batch, key=lambda s: s[0].shape[0], reverse=True)
    max_data_length = batch[0][0].shape[0]
    batch_size = len(batch)
    # 以最大的长度创建0张量
    inputs = np.zeros((batch_size, max_data_length), dtype='int64')
    labels = np.zeros((batch_size, max_data_length), dtype='int64')
    indices = np.arange(batch_size).tolist()
    random.shuffle(indices)
    for x in indices:
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.shape[0]
        label_length = target.shape[0]
        # 输入文本数据的参数和标签长度要一样的
        assert seq_length == label_length
        # 将数据插入都0张量中，实现了padding
        inputs[x, :seq_length] = tensor[:]
        labels[x, :label_length] = target[:]
    return torch.from_numpy(inputs), torch.from_numpy(labels)
