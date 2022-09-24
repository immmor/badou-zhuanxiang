# -*- coding: utf-8 -*-

import re
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from preprocess import words_exist

"""
数据加载
"""
class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.index_to_label = {0: '好评', 1: '差评'}
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        self.config["class_num"] = len(self.index_to_label)
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.load()


    def load(self):
        self.data = []
        df = pd.read_csv(self.path)
        for label, review in zip(df['label'], df['review']):
            if self.config["model_type"] == "bert":
                input_id = self.tokenizer.encode(review, max_length=self.config["max_length"], pad_to_max_length=True)
            elif self.config['word_seg']:
                input_id = self.encode_sentence_by_word(review)
            else:
                input_id = self.encode_sentence_by_char(review)
            input_id = torch.LongTensor(input_id)
            label_index = torch.LongTensor([label])
            self.data.append([input_id, label_index])

    def encode_sentence_by_word(self, text):
        input_id = []
        word_compiler = re.compile(self.config['word_pat'])
        for word in text.split(' '):
            if words_exist(word, word_compiler):
                sub_words = list(word)
                for sword in sub_words:
                    input_id.append(self.vocab.get(sword, self.vocab["[UNK]"]))
            else:
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    def encode_sentence_by_char(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict


#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("valid_set.csv", Config)
    print(dg[1])
