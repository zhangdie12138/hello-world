# -*- coding: utf-8 -*-

from load_data import *
from word_vector import *
from preprocessing_data import *
import numpy as np 
from Model_Trainer import *
import time
from partial_fit import *

if '__main__' == __name__:
    #得到抽取的样本内容和结果标签并存储
    
    label_messages_file="D:/zd/webdata mining/Data/label-messages.txt"
    num, mess_Po, mess_Ne = load_message(label_messages_file)
    sample_file="D:/zd/webdata mining/Data/sample-messages.txt"
    # size控制选择的样本数量
    data_selection(mess_Po, mess_Ne,sample_file,size= 1)
    content,label=load_samples(sample_file)
    content_file="D:/zd/webdata mining/Data/content.txt"
    label_file="D:/zd/webdata mining/Data/label.txt"
    data_storage(content_file, label_file,content, label)

    #生成tf_idf的词向量（去部分停用词）并存储
    stopwords_file="D:/zd/webdata mining/Data/stopwords.txt"
    fenci_file="D:/zd/webdata mining/Data/content_fenci.txt"
    stopwords=loadStopWords(stopwords_file)
    fenci(content_file,fenci_file,stopwords)
    


    #数据预处理
    content = []
    with open('D:/zd/webdata mining/Data/content_fenci.txt', 'r',encoding='utf-8') as f:
        for line in f.readlines():  
            line=line.strip('\n')
            content.append(line)
    
    label = []
    with open('D:/zd/webdata mining/Data/label.txt', 'r') as f:
        for line in f.readlines():  
            line=line.strip('\n')
            label.append(int(line)) 

    training_data, test_data, training_target, test_target = split_data(content, label)

    #增量训练
    results_file="D:/zd/webdata mining/Data/results.txt"
    fit = Partial_fit(training_data,training_target)
    results=fit.partial_fit(test_data,test_target,results_file)
    fit.plot(*results)




