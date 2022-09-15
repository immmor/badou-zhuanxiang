#week4作业

'''
1.在kmeans聚类基础上，实现根据类内距离排序，输出结果
2.在不进行文本向量化的前提下对文本进行kmeans聚类(可采用jaccard距离)
'''

import jieba
import numpy as np
import random
import sys
import json
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict


class SortKMeans:
    def __init__(self,textpath,cluster_num,isVec,isCosine):#isVec 为一个bool参数,表明是否选用词向量模式
        self.isVec=isVec
        self.isCosine=isCosine
        self.textpath=textpath
        self.sentences=self.load_sentence()
        self.cluster_num=cluster_num
        self.points=self.__pick_start_point(self.sentences,cluster_num)
        self.buffer={}#因为过程中会有许多重复的距离计算,做一个缓存字典

    def cluster(self):
        if self.isVec==False:
            result=[]
            for i in range(self.cluster_num):
                result.append([])
            for item in self.sentences:
                distance_min=sys.maxsize
                index=-1
                for i in range(len(self.points)):
                    distance=self.__distance(item,self.points[i])
                    if distance<distance_min:
                        distance_min=distance
                        index=i
                result[index]=result[index]+[item]
            new_center=[]
            distances=[]
            for item in result:
                center,distance_to_all=self.__center(item)
                new_center.append(center)
                distances.append(distance_to_all)
            #中心点未改变,说明到达稳态,阶数递归
            if (self.points==new_center):
                return result
            self.points=new_center
            return self.cluster()
        else:
            #加载词向量模型
            model=self.load_word2vec_model("model.w2v")
            #加载所有标题
            sentences=self.load_sentence()
            #将所有标题向量化
            vectors=self.sentence_to_vectors(model)
            #定义一个KMeans计算类
            kmeans=KMeans(self.cluster_num)
            #进行聚类的计算
            kmeans.fit(vectors)

            sentence_label_dict=defaultdict(list)
            for sentence,label in zip(sentences,kmeans.labels_):#取出句子和标签
                sentence_label_dict[label].append(sentence)#将同标签的句子放到一起
            #计算类内距离
            density_dict=defaultdict(list)
            for vector_index,label in enumerate(kmeans.labels_):
                vector=vectors[vector_index]#某句话的向量
                center=kmeans.cluster_centers_[label]#对应类别的中心向量
                if self.isCosine==True:
                    distance=self.cosine_distance(vector,center)#计算距离
                else:
                    distance=self.eculid_distance(vector,center)
                density_dict[label].append(distance)
            for label,distance_list in density_dict.items():
                density_dict[label]=np.mean(distance_list)
            density_order=sorted(density_dict.items(),key=lambda x:x[1],reverse=True)#按照平均距离排序,向量夹角余弦值越接近1,距离越小

            #按照余弦距离输出
            for label,distance_avg in density_order:
                print("clusters %s avg distance %f:"%(label,distance_avg))
                sentences=sentence_label_dict[label]
                for i in range(len(sentences)):
                    print(sentences[i].replace(" ",""))
                print("---------------------------------")

    def cosine_distance(self,vec1,vec2):
        vec1=vec1/np.sqrt(np.sum(np.square(vec1)))
        vec2=vec2/np.sqrt(np.sum(np.square(vec2)))
        return np.sum(vec1*vec2)

    def eculid_distance(self,vec1,vec2):
        return np.sqrt(np.sum(np.square(vec1-vec2)))

    #选取新的中心的方法
    #由于使用的是离散的字符串,所以无法通过原有的平均方式计算新的中心
    #人为设定新中心更替方式:
    #选取类别中到其他所有点距离总和最短的字符串为中心
    def __center(self,cluster_sentences):
        center="              "#设置一个不存在的站位字符
        distance_to_all=999999999#站位最大距离
        for sentence_a in cluster_sentences:
            distance=0
            for sentence_b in cluster_sentences:
                distance+=self.__distance(sentence_a,sentence_b)
            distance/=len(cluster_sentences)
            if distance<distance_to_all:
                center=sentence_a
                distance_to_all=distance
        return center,distance_to_all
    #将距离函数替换为非向量算法
    #此处使用jaccard距离
    #使用字典缓存加快距离计算
    def __distance(self,p1,p2):
        if p1+p2 in self.buffer:
            return self.buffer[p1+p2]
        elif p2+p1 in self.buffer:
            return self.buffer[p2+p1]
        else:
            #jaccard距离:公共词越多越相近
            distance=1-len(set(p1)&set(p2))/len(set(p1).union(set(p2)))
            self.buffer[p1+p2]=distance
            return distance
    #随机选取初始点,改成随机挑选字符串
    def __pick_start_point(self,sentences,cluster_num):
        return random.sample(sentences,cluster_num)
    #加载数据集
    def load_sentence(self):
        if self.isVec==True:
            sentences=set()
            with open(self.textpath,encoding="utf8") as f:
                for line in f:
                    sentence=line.strip()
                    sentences.add(" ".join(jieba.cut(sentence)))
            print("获取句子数量:",len(sentences))
            return sentences
        else:
            sentences=[]
            with open(self.textpath,encoding="utf8") as f:
                for index,line in enumerate(f):
                    sentences.append(line.strip())
            return sentences

    #输入模型文件路径,加载训练好的模型
    def load_word2vec_model(self,path):
        model=Word2Vec.load(path)
        return model

    #将文本向量化
    def sentence_to_vectors(self,model):
        vectors=[]
        for sentence in self.sentences:#sentence是分好词的,空格分开
            words=sentence.split()
            vector=np.zeros(model.vector_size)
            #所有词的向量相加求平均,作为句子向量
            for word in words:
                try:
                    vector+=model.wv[word]
                except KeyError:
                    #部分词在训练中未出现,用全0向量代替
                    vector+=np.zeros(model.vector_size)
            vectors.append(vector/len(words))
        return np.array(vectors)
if __name__=="__main__":
    sortkmean=SortKMeans("titles.txt",10,True,True)
    res=sortkmean.cluster()
    print(json.dumps(res,ensure_ascii=False,indent=2))