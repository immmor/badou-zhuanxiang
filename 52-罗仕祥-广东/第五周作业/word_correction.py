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
        self.language_model = language_model


    def load_tongyin_dict(self, path):
        tongyin_dict = {}
        for i in open(path, encoding="utf8").readlines():
            k = i.split()[0]
            v = i.split()[1]
            tongyin_dict[k] = v
        return tongyin_dict

    def get_the_change_sorce(self, tongyin, char_list, key):

        if len(tongyin) == 0:
            return [99999]
        else:
            result = []
            for i in tongyin:
                char_list[key] = i
                sentence = "".join(char_list) # 替换后的语句
                sentence_final_score = self.language_model.calc_sentence_ppl(sentence)
                sentence_final_score -= self.sentence_original_score
                # print(i, sentence_final_score)
                result.append(sentence_final_score)
            return result

    def correction(self, string):

        self.sentence_original_score = self.language_model.calc_sentence_ppl(string)    # 原始语句ppl得分
        # print("原始语句ppl得分： ", self.sentence_original_score)
        fix_list = []
        char_list = list(string)
        tongyin_dict = self.load_tongyin_dict(path="tongyin.txt")
        for key, value in enumerate(char_list):
            # 替换每一个字
            tongyin = tongyin_dict.get(value, [])
            sentence_score_list = self.get_the_change_sorce(tongyin, copy.deepcopy(char_list), key)
            fix_list += [char_list[key]]
            if min(sentence_score_list) < 0:
                sub_char = tongyin[sentence_score_list.index(min(sentence_score_list))]
                print("第%d个建议修改：%s -> %s, 概率提升： %f" %(key, value, sub_char, min(sentence_score_list)))
                fix_list[-1] = sub_char
        return "".join(fix_list)


corpus = open("财经.txt", encoding="utf8").readlines()
lm = NgramLanguageModel(corpus, n=3)
cr = Corrector(lm)
string = "每国货币政册空间不大"
fix_string = cr.correction(string)

print("修改前：", string)
print("修改后：", fix_string) #美国货币政策空间不大