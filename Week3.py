#week3作业

#目标输出;顺序不重要
# target = [
#     ['经常', '有意见', '分歧'],
#     ['经常', '有意见', '分', '歧'],
#     ['经常', '有', '意见', '分歧'],
#     ['经常', '有', '意见', '分', '歧'],
#     ['经常', '有', '意', '见分歧'],
#     ['经常', '有', '意', '见', '分歧'],
#     ['经常', '有', '意', '见', '分', '歧'],
#     ['经', '常', '有意见', '分歧'],
#     ['经', '常', '有意见', '分', '歧'],
#     ['经', '常', '有', '意见', '分歧'],
#     ['经', '常', '有', '意见', '分', '歧'],
#     ['经', '常', '有', '意', '见分歧'],
#     ['经', '常', '有', '意', '见', '分歧'],
#     ['经', '常', '有', '意', '见', '分', '歧']
# ]

#词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
Dict = {"经常":0.1,
        "经":0.05,
        "有":0.1,
        "常":0.001,
        "有意见":0.1,
        "歧":0.001,
        "意见":0.2,
        "分歧":0.2,
        "见":0.05,
        "意":0.05,
        "见分歧":0.05,
        "分":0.1}
#待切分文本
sentence = "经常有意见分歧"

#实现全切分
#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
        # 寻找字典中的最长词
        max_length = 0
        for _, value in enumerate(Dict):
                if len(value) > max_length:
                        max_length = len(value)

        # 寻找分词路径中的首词
        # 首先输出全切分
        target_temp = []
        temp = []
        j = 0
        for i in range(0, len(sentence)):
                while j < max_length:
                        if i + j < len(sentence):
                                window = sentence[i:i + j + 1]
                        else:
                                window = ''
                        if window in Dict.keys():
                                target_temp.append(window)
                        j += 1
                j = 0
        print(target_temp)
        return target_temp
#寻找词首,输出当前位置前的字符串
def findFirstWord(all_cut,sentence):


if __name__ == "__main__":
        all_cut(sentence, Dict)