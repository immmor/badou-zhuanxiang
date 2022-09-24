# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "input_data/train_hw.csv",
    "valid_data_path": "input_data/test_hw.csv",
    "vocab_path": "../chars.txt",
    "model_type": "gated_cnn",
    "max_length": 20,
    "hidden_size": 128,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 15,
    "batch_size": 64,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"pretrain_model\chinese-bert_chinese_wwm_pytorch",
    "seed": 987,
    "class_num": 2
}