# -*- coding: utf-8 -*-

"""
数据预处理
"""
import re
import jieba
import pandas as pd
from sklearn.model_selection import train_test_split


def jieba_tok(line):
    return ' '.join(list(jieba.cut(line, cut_all=True)))


def words_exist(word, compiler):
    if compiler.search(word):
        return True
    return False


def get_word_vocab_set(train_set):
    train_vocab_set = set()
    zh_com = re.compile("[\u4e00-\u9fa5]")
    for review_tok in train_set['review_tok']:
        review_tok_list = review_tok.split(' ')
        for tok in review_tok_list:
            if words_exist(tok, zh_com):
                train_vocab_set.add(tok)
            else:
                tok_list = list(tok)
                for sub_tok in tok_list:
                    train_vocab_set.add(sub_tok)
    return train_vocab_set


def get_char_vocab_set(train_set):
    train_vocab_set = set()
    for review in train_set['review']:
        for char in list(review):
            train_vocab_set.add(char)
    return train_vocab_set


if __name__ =='__main__':
    # 数据分割
    df = pd.read_csv('文本分类练习.csv')
    df['review_tok'] = df['review'].apply(jieba_tok)

    x, y = df[['review_tok', 'review']], df['label']
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.1, random_state=42)
    train_set = pd.DataFrame({'label': y_train, 'review_tok': x_train['review_tok'], 'review': x_train['review']})
    valid_set = pd.DataFrame({'label': y_valid, 'review_tok': x_valid['review_tok'], 'review': x_valid['review']})

    train_set.to_csv('train_set.csv', index=False, columns=['label', 'review'])
    valid_set.to_csv('valid_set.csv', index=False, columns=['label', 'review'])

    train_vocab_set = get_char_vocab_set(train_set)

    with open('vocab.txt', mode='w', encoding='utf-8') as vocab_file:
        print('\n'.join(sorted(train_vocab_set)), file=vocab_file)
