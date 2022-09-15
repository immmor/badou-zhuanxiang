import copy
import math
from collections import defaultdict

class NgramLanguageModel:
    def __init__(self,corpus=None,n=3):
        self.gram_n=n
        self.sep="\'"
        self.sos='<sos>'
        self.eos="<eos>"
        self.unk_prob=1e-6
        self.backoff_prob=0.4
        self.ngram_count_dict=dict((x+1,defaultdict(int)) for x in range(n))
        self.ngram_count_prob_dict=dict((x+1,defaultdict(int)) for x in range(n))
        self.ngram_count(corpus)
        self.calc_ngram_prob()
    #cut the text into character or word or token
    def sentence_segment(self,sentence):
        return list(sentence)
    #count the number of the ngram
    def ngram_count(self,corpus):
        for sentence in corpus:
            word_lists=self.sentence_segment(sentence)
            word_lists=[self.sos]+word_lists+[self.eos]
            for windowsize in range(1,self.gram_n+1):
                for index,word in enumerate(word_lists):
                    #when arrive at the end of the word_list,maybe the number of characters less than windowsize,then pass
                    if len(word_lists[index:index+windowsize]) !=windowsize:
                        continue
                    #use the sep join the words construct a ngram and save it
                    ngram=self.sep.join(word_lists[index:index+windowsize])
                    self.ngram_count_dict[windowsize][ngram]+=1
        #count the total number of the words,and using it to compute the first ngram prob
        self.ngram_count_dict[0]=sum(self.ngram_count_dict[1].values())
        return
    #coumpute the ngram's prob
    def calc_ngram_prob(self):
        for windowsize in range(1,self.gram_n+1):
            for ngram,count in self.ngram_count_dict[windowsize].items():
                if windowsize>1:
                    ngram_splits=ngram.split(self.sep)
                    ngram_prefix=self.sep.join(ngram_splits[:-1])
                    ngram_prefix_count=self.ngram_count_dict[windowsize-1][ngram_prefix]
                else:
                    ngram_prefix_count=self.ngram_count_dict[0]
                self.ngram_count_prob_dict[windowsize][ngram]=count/ngram_prefix_count
        return
    #get the ngram prob , in the process the fall back smooth is used ,the value is fixed
    def get_ngram_prob(self,ngram):
        n=len(ngram.split(self.sep))
        if ngram in self.ngram_count_prob_dict[n]:
            return self.ngram_count_prob_dict[n][ngram]
        elif n==1:
            return self.unk_prob
        else:
            ngram=self.sep.join(ngram.split(self.sep)[1:])
            return self.backoff_prob*self.get_ngram_prob(ngram)
    #give the prob of the sentence
    def predict(self,sentence):
        word_list=self.sentence_segment(sentence)
        word_list=[self.sos]+word_list+[self.eos]
        sentence_prob=0
        for index,word in enumerate(word_list):
            ngram=self.sep.join(word_list[max(0,index-self.gram_n+1):index+1])
            prob=self.get_ngram_prob(ngram)
            sentence_prob+=math.log(prob)
        return sentence_prob

class TextVeryfication:
    def __init__(self,language_model):
        self.language_model=language_model
        #Candidate dict
        self.candidate_dict=self.load_candidate_dict("tongyin.txt")
        self.threshold=7
    def load_candidate_dict(self,path):
        homephone_dict={}
        with open(path,encoding="utf8") as f:
            for line in f:
                word,homephone=line.split()
                homephone_dict[word]=list(homephone)
        return homephone_dict
    def get_candidate_sentence_prob(self,candidates,char_list,index):
        if candidates==[]:
            return [-1]
        result=[]
        for char in candidates:
            char_list[index]=char
            sentence="".join(char_list)
            sentence_prob=self.language_model.predict(sentence)
            sentence_prob-=self.sentence_baseline_prob
            result.append(sentence_prob)
        return result
    def veryFication(self,string):
        char_list=list(string)
        fix={}
        self.sentence_baseline_prob=self.language_model.predict(string)
        for index,char in enumerate(char_list):
            candidates=self.candidate_dict.get(char,[])
            candidates_probs=self.get_candidate_sentence_prob(candidates,copy.deepcopy(char_list),index)
            if max(candidates_probs)>self.threshold:
                sub_char=candidates[candidates_probs.index(max(candidates_probs))]
                print("第%d个字建议修改:%s->%s,概率提升:%f"%(index,char,sub_char,max(candidates_probs)))
                fix[index]=sub_char
        char_list=[fix[i] if i in fix else char for i,char in enumerate(char_list)]
        return "".join(char_list)


if __name__=="__main__":
    corpus=open("corpus\财经.txt",encoding="utf8").readlines()
    lm=NgramLanguageModel(corpus,3)
    textVeryfication=TextVeryfication(lm)
    sentence="每国货币政册空间不大,金年亩亲节不仅是各大熵家粗销的阵的"
    correct_sentence=textVeryfication.veryFication(sentence)
    print("fix before:",sentence)
    print("fix after:",correct_sentence)
