import json
import copy
from ngram_language_model import NgramLanguageModel
"""
文本纠错demo
加载同音字字典
加载语言模型
基本原理：
对于文本中每一个字，判断在其同音字中是否有其他字，在替换掉该字时，能使得语言模型计算的成句概率提高
"""


class Corrector:
    def __init__(self, language_model):
        pass

    def correction(self, string):
        pass

cr = Corrector(lm)
string = "每国货币政册空间不大"
fix_string = cr.correction(string)
print("修改前：", string)
print("修改后：", fix_string) #美国货币政策空间不大