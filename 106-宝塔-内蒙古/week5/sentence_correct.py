from language_model import NgramLanguageModel

"""
基于语言模型的文本纠错
"""
class SentenceCorrect:
    def __init__(self, language_model, phone_dict_file, prob_improve=5):
        self.language_model = language_model    # 语言模型
        self.prob_improve = prob_improve    # prob提高值
        self.phone_dict = dict()    # 加载易错字数据集
        for dict_line in phone_dict_file.readlines():
            dict_line_list = dict_line.replace('\n', '').split(' ')
            self.phone_dict[dict_line_list[0]] = dict_line_list[1]

    # 文本纠错
    def correct(self, sentence):
        orig_score = self.language_model.predict(sentence)    # 计算当前得分
        best_sentence = sentence
        best_score = orig_score
        best_imporve = 0
        # 将所有字替换为对应的易错字，并计算得分
        for i in range(len(sentence)):
            if sentence[i] in self.phone_dict:
                for word in self.phone_dict[sentence[i]]:
                    # 替换易错字，计算得分
                    tmp_sentence = best_sentence[0:i] + word + best_sentence[i+1:]
                    tmp_score = self.language_model.predict(tmp_sentence)
                    tmp_imporve = tmp_score - orig_score
                    # 当前prob提高值大于最好值时，替换最好值
                    if tmp_imporve > self.prob_improve and tmp_imporve > best_imporve:
                        best_sentence = tmp_sentence
                        best_score = tmp_score
                        best_imporve = tmp_imporve
        return best_sentence, best_score

if __name__ == '__main__':
    sentence = "每国货币政册空间不大"
    with open('my_model.txt', encoding='utf8', mode='r') as model_file, \
         open('tongyin.txt', encoding='utf8', mode='r') as phone_dict_file:
        language_model = NgramLanguageModel(3, model_file, 0.3)
        sentence_correct = SentenceCorrect(language_model, phone_dict_file)
        sentence_fixed, score = sentence_correct.correct(sentence)
        print('{}\n{}\t{:.4f}'.format(sentence, sentence_fixed, score))
