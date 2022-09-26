import math

"""
Ngram语言模型
"""
class NgramLanguageModel:
    def __init__(self, ngram, model_file=None, backoff_factor=0.4):
        self.ngram = ngram    # n
        self.bos = "<bos>"    # begin of sentence
        self.eos = "<eos>"    # end of sentence
        self.backoff_factor = backoff_factor    # 回退平滑方法时使用的系数
        self.word_dict = dict()    # 语言模型
        self.all_words_count = 0    # 语料中的所有字数
        if model_file is not None:
            self.load_model_file(model_file)

    # 读取语言模型文件
    def load_model_file(self, model_file):
        for ngram_line in model_file.readlines():
            ngram_list = ngram_line.replace('\n', '').split('\t')
            self.word_dict[ngram_list[0]] = int(ngram_list[1])
            self.all_words_count += int(ngram_list[1])

    # 根据语料生成Ngram语言模型
    def fit(self, corpus_file, save_file=None):
        sentences = corpus_file.readlines()
        for sentence in sentences:
            sentence_list = [self.bos] + list(sentence.replace('\n', '')) + [self.eos]
            for i in range(len(sentence_list)):
                for j in range(1, self.ngram + 1):
                    if i+j > len(sentence_list):    # 规避超出句子长度时的重复计数问题
                        continue
                    ngram = ''.join(sentence_list[i:i + j])
                    if ngram not in self.word_dict:
                        self.word_dict[ngram] = 0
                    self.word_dict[ngram] += 1    # ngram计数加一
                    self.all_words_count += 1    # 全部单词计数加一

        # 保存模型文件
        if save_file is not None:
            for ngram in self.word_dict:
                print('{}\t{}'.format(ngram, self.word_dict[ngram]), file=save_file)

    # 计算ngram概率
    def get_ngram_prob(self, ngram):
        ngram_n = len(ngram)
        ngram_line = ''.join(ngram)
        # ngram在模型中时，直接计算
        if ngram_line in self.word_dict:
            pre_ngram_line = ''.join(ngram[:-1])
            pre_count = self.word_dict[pre_ngram_line] if ngram_n > 1 else self.all_words_count
            return self.word_dict[ngram_line] / pre_count
        # ngram不在模型中时，回退平滑
        else:
            # 未知单字时，取平均值
            if ngram_n == 1:
                return 1 / self.all_words_count
            # 未知ngram时，回退平滑
            else:
                return self.backoff_factor * self.get_ngram_prob(ngram[1:])

    # 计算句子概率
    def predict(self, sentence):
        sentence_list = [self.bos] + list(sentence.replace('\n', '')) + [self.eos]
        sentence_prob = 0
        for i in range(len(sentence_list) - self.ngram + 1):
            ngram = sentence_list[i : i+self.ngram]
            ngram_prob = self.get_ngram_prob(ngram)
            # print(ngram, ngram_prob)
            sentence_prob += math.log(ngram_prob)
        return sentence_prob

if __name__ == '__main__':
    sentence = "萦绕在世界经济的阴霾仍未消退"
    with open('my_model.txt', encoding='utf8', mode='r') as model_file:
        my_lm = NgramLanguageModel(3, model_file, 0.3)
        prob = my_lm.predict(sentence)
        print('{}\t{:.4f}'.format(sentence, prob))
    # my_lm = NgramLanguageModel(3)
    # with open('财经.txt', encoding='utf8', mode='r') as corpus_file:
    #     with open('my_model.txt', encoding='utf8', mode='w') as save_file:
    #         my_lm.fit(corpus_file, save_file)
