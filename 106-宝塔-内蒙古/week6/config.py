# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "train_set.csv",
    "valid_data_path": "valid_set.csv",
    "vocab_path":"vocab.txt",
    "model_type":"cnn",
    "max_length": 30,
    "hidden_size": 128,
    "kernel_size": 3,
    "num_layers": 6,
    "epoch": 5,
    "batch_size": 256,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "word_pat": "[\u4e00-\u9fa5]",
    "word_seg": False,
    "pretrain_model_path":r"D:\badou\pretrain_model\chinese-bert_chinese_wwm_pytorch",
    "seed": 42
}