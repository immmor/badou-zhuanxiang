#week4作业

'''

1.在kmeans聚类基础上，实现根据类内距离排序，输出结果

2.在不进行文本向量化的前提下对文本进行kmeans聚类(可采用jaccard距离)
'''

import jieba
import math
import sys
import random
import numpy as np

#计算公共词
#前面1-的原因是使得文本相似是距离接近0，文本不同时文本接近1，更接近距离的概念
def jaccard_distance(list_of_words1, list_of_words2):
    return 1 - len(set(list_of_words1) & set(list_of_words2)) / len(set(list_of_words1) | set(list_of_words2))

def load_sentence(path):
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    sentences = list(sentences)
    format_sentences = list()
    for item in sentences:
        format_sentences.append(item.split(" "))
    print("获取句子数量：", len(format_sentences))
    return format_sentences

class KMeansClusterer:  # k均值聚类
    def __init__(self, ndarray, cluster_num):
        self.ndarray = ndarray
        self.cluster_num = cluster_num
        self.points = self.__pick_start_point(ndarray, cluster_num)
        self.dist_inner = []

    def cluster(self):
        result = []
        for i in range(self.cluster_num):
            result.append([])
        for item in self.ndarray:
            distance_min = sys.maxsize
            index = -1
            for i in range(len(self.points)):
                distance = self.__distance(item, self.points[i])
                if distance < distance_min:
                    distance_min = distance
                    index = i
            result[index] = result[index] + [item]
        new_center = []
        new_dist_inner = []
        for item in result:
            class_center, class_dist = self.__center(item)
            new_center.append(class_center)
            new_dist_inner.append(class_dist)
        # 中心点未改变，说明达到稳态，结束递归
        if sorted(self.points) == sorted(new_center):
            sum = self.__sumdis(result)
            return result, self.points, self.dist_inner, sum
        self.points = new_center
        self.dist_inner = new_dist_inner
        return self.cluster()

    def __sumdis(self,result):
        #计算总距离和
        sum = 0
        for i in range(len(self.points)):
            for j in range(len(result[i])):
                sum += self.__distance(result[i][j], self.points[i])
        return sum

    def __center(self, list):
        min_i = -1
        min_dist = sys.maxsize
        for i in range(0, len(list)):
            dist_in = 0
            for j in range(0, len(list)):
                dist_in += self.__distance(list[i], list[j])
            if dist_in < min_dist:
                min_dist = dist_in
                min_i = i
        min_dist /= len(list)
        return list[min_i], min_dist

    def __distance(self, p1, p2):
        return jaccard_distance(p1, p2)

    def __pick_start_point(self, ndarray, cluster_num):
        if cluster_num < 0 or cluster_num > len(ndarray):
            raise Exception("簇数设置有误")
        # 取点的下标
        indexes = random.sample(np.arange(0, len(ndarray), step=1).tolist(), cluster_num)
        points = []
        for index in indexes:
            points.append(ndarray[index])
        return np.array(points)

def main():
    sentences = load_sentence("titles.txt")  # 加载所有标题
    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    kmeans = KMeansClusterer(sentences, n_clusters)
    result, centers, dist_inner, distances = kmeans.cluster()
    dist_dict = dict()
    for idx, dist in enumerate(dist_inner):
        dist_dict.update({idx: dist})
    dist_sorted = sorted(dist_dict.items(), key=lambda x:x[1])
    for item in dist_sorted:
        print(item)
        print(result[item[0]])
    #print(result)
    #print(centers)
    #print(distances)

if __name__ == '__main__':
    main()
