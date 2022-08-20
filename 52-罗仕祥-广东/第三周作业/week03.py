#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2022/8/18 20:24
@Author  : luoshixiang
@Email   : just_rick@163.com
@File    : week03.py
@effect  : 实现基于词表的全切分
"""

import json
import jieba
import itertools


def cut_method3(string, prefix_dict):
    if string == "":
        return []
    words = []  # 准备用于放入切好的词
    start_index, end_index = 0, 1  #记录窗口的起始位置
    window = string[start_index:end_index] #从第一个字开始
    find_word = window  # 将第一个字先当做默认词
    while start_index < len(string):
        #窗口没有在词典里出现
        if window not in prefix_dict or end_index > len(string):
            words.append(find_word)  #记录找到的词
            start_index += len(find_word)  #更新起点的位置
            end_index = start_index + 1
            window = string[start_index:end_index]  #从新的位置开始一个字一个字向后找
            find_word = window
        #窗口是一个词,且不是任何词的前缀
        elif prefix_dict[window] == 1:
            words.append(window)  # 记录找到的词
            start_index += len(window)  # 更新起点的位置
            end_index = start_index + 1
            window = string[start_index:end_index]  # 从新的位置开始一个字一个字向后找
            find_word = window
        #窗口是一个词，但是有包含它的词，所以要再往后看
        elif prefix_dict[window] == 2:
            find_word = window  #查找到了一个词，还要在看有没有比他更长的词
            end_index += 1
            window = string[start_index:end_index]
        #窗口是一个前缀
        elif prefix_dict[window] == 0:
            end_index += 1
            window = string[start_index:end_index]
    #最后找到的window如果不在词典里，把单独的字加入切词结果
    if prefix_dict.get(window) != 1:
        words += list(window)
    else:
        words.append(window)
    return words

def load_prefix_word_dict(path):
    # 根据字典建立state dict
    prefix_dict = {}
    with open(path, 'r', encoding='utf-8') as fr:
        json_file = json.load(fr)  # 若文件不为空但json_file读出来为空，注意编码格式是否匹配
        for word,frequency in json_file.items():    # 获取词与词频
            for i in range(1, len(word)):
                if word[:i] not in prefix_dict:  # 不能用前缀覆盖词
                    prefix_dict[word[:i]] = 0  # 前缀
                if prefix_dict[word[:i]] == 1:
                    prefix_dict[word[:i]] = 2
            prefix_dict[word] = 1  # 词
    return prefix_dict

def get_dag(sentence, prefix_dict):
    DAG = {}
    N = len(sentence)
    for i in range(N):
        tmplist = []
        j = i
        frag = sentence[i]
        while j < N and frag in prefix_dict:
            if prefix_dict[frag]:
                tmplist.append(j)
            j += 1
            frag = sentence[i:j + 1]
        if not tmplist:
            tmplist.append(i)
        DAG[i] = tmplist
    return DAG

def get_comb(DAG, sentence):
    Dict = [v for k, v in DAG.items()]   # 取出字典中的值组成列表
    combination = [list(i) for i in list(itertools.product(*Dict))]  # 每个列表抽取1个数据的所有组合

    tmp_combination = []
    for i in range(len(combination)):
        b = combination[i]
        tmp = []
        for j in range(len(b)):
            if b[j] not in tmp:
                if j == 0:
                    tmp.append(b[j])
                elif b[j] > b[j - 1]:
                    tmp.append(b[j])
                elif b[j] <= b[j - 1]:
                    del tmp[-1]
                    tmp.append(b[j])
                else:
                    tmp.append(b[j])
        tmp_combination.append(tmp)

    cc = []
    [cc.append(i) for i in tmp_combination if i not in cc]     # 对 d 去重，存到列表 e 中

    d = []
    for i in range(len(cc)):
        g = cc[i]
        f = []
        for j in range(len(g)):
            if j == 0:
                f.append(sentence[:g[j] + 1])
            else:
                f.append(sentence[g[j - 1] + 1:g[j] + 1])
        d.append(f)


    word_list = [k for k,v in prefix_dict.items() if v!=0]
    # 剔除不存在的词组的组合
    e = []
    for i in d:
        k = []
        for j in i:
            if j not in word_list:
                k = []
                break
            else:
                k.append(j)
        e.append(k)
    target = [i for i in e if i]     # 去重
    return target

if __name__ == '__main__':
    # 待切分文本
    sentence = "经常有意见分歧"
    path = 'dict.json'  # 词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改

    # 1.根据字典建立state dict
    prefix_dict = load_prefix_word_dict(path)

    # 2.DAG 获得有向无环图
    DAG = get_dag(sentence, prefix_dict)

    # 3. 从DAG中输出所有可能的词汇
    target = get_comb(DAG, sentence)

    for i in target:
        print(i)

