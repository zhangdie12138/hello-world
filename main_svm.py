# -*- coding: utf-8 -*-

from load_data import *
from word_vector import *
from preprocessing_data import *
import numpy as np 
from Model_Trainer import *
import time
from Model_Predictor import *


def stream_data(content):
  content=content.tocsr()
  for i in range(content.shape[0]):
    yield content[i]

def get_minibatch(stream,size):
  X_train = []
  x_train=[]
  cur_line_num = 0
  try:
    for i in range(size):
      temp = next(stream)
      X_train.append(temp.toarray())
    x_train=X_train[0]
    for i in range(size-1):
      x_train=np.vstack((x_train,X_train[i+1]))     
  except StopIteration:
    pass
  return x_train

def iter_minibatches(stream,minibatch_size=1000):
  X_train = get_minibatch(stream,minibatch_size)
  while len(X_train):
    yield X_train
    X_train = get_minibatch(stream,minibatch_size)

if '__main__' == __name__:
    label_messages_file="D:/zd/webdata mining/Data/label-messages.txt"
    num, mess_Po, mess_Ne = load_message(label_messages_file)
    sample_file="D:/zd/webdata mining/Data/sample-messages.txt"
    data_selection(mess_Po, mess_Ne,sample_file,size=1000)
    content,label=load_samples(sample_file)
    content_file="D:/zd/webdata mining/Data/content.txt"
    label_file="D:/zd/webdata mining/Data/label.txt"
    data_storage(content_file, label_file,content, label)

    #生成tf_idf的词向量（去部分停用词）并存储
    stopwords_file="D:/zd/webdata mining/Data/stopwords.txt"
    fenci_file="D:/zd/webdata mining/Data/content_fenci.txt"
    stopwords=loadStopWords(stopwords_file)
    fenci(content_file,fenci_file,stopwords)
    word_vector_file="D:/zd/webdata mining/Data/word_vector.mtx"
    tf_idf(fenci_file,word_vector_file)


    #数据预处理
    content = io.mmread('D:/zd/webdata mining/Data/word_vector.mtx')
    label = []
    with open('D:/zd/webdata mining/Data/label.txt', 'r') as f:
        for line in f.readlines():  
            line=line.strip('\n')
            label.append(int(line)) 
    	

    training_data, test_data, training_target, test_target = split_data(content, label)

    #不使用增量训练
    
    training_data, test_data = dimensionality_reduction(training_data.toarray(), test_data.toarray(), type='pca')
    
    model_file=["D:/zd/webdata mining/Data/SVM_linear_model.pkl","D:/zd/webdata mining/Data/SVMRbf_model.pkl","D:/zd/webdata mining/Data/bayes_model.pkl"]  

    fit_svmlin = SVMLinear(training_data,array(training_target))
    fit_svmlin.train_classifier(model_file[0])
    
    fit_svmrbf = SVMRbf(training_data,array(training_target))
    fit_svmrbf.train_classifier(model_file[1])

    fit_bayes = Trainer_bayes(training_data.toarray(),array(training_target))
    fit_bayes.train_classifier(model_file[2])
    
    models=["SVMLinear","SVMRbf","bayes"]
    f = open("D:/zd/webdata mining/Data/reports.txt",'w')
    for i in range(2):
        clf = joblib.load(model_file[i])
        pre=Predictor(clf)
        reports=pre.sample_predict(test_data,array(test_target))
        f.write(models[i]+'\n')
        f.write(reports[0])
        f.write(str(reports[1])+'\n')

    clf = joblib.load(model_file[2])
    pre=Predictor(clf)
    reports=pre.sample_predict(test_data.toarray(),array(test_target))
    f.write(models[2]+'\n')
    f.write(reports[0])
    f.write(str(reports[1])+'\n')



    #预测未标签数据
    nolabel_messages_file="D:/zd/webdata mining/Data/nolabel_messages.txt"
    #生成tf_idf的词向量（去部分停用词）并存储
    fenci_file="D:/zd/webdata mining/Data/nolabel_fenci.txt"
    stopwords=loadStopWords(stopwords_file)
    fenci(nolabel_messages_file,fenci_file,stopwords)
    word_vector_file="D:/zd/webdata mining/Data/nolabel_wordvector.mtx"
    tf_idf(fenci_file,word_vector_file)
    messsages = io.mmread(word_vector_file)
    stream = stream_data(messsages)
    minibatch_iterators = iter_minibatches(stream,minibatch_size= 5000)
    clf = joblib.load("D:/zd/webdata mining/Data/SVM_linear_model.pkl")
    result_file = "D:/zd/webdata mining/Data/SVMLinear_pre.txt"
    fr = open(result_file,'w')
    for i, X_train in enumerate(minibatch_iterators):
        pre = Predictor(clf)
        clf.new_predict(X_train,fr)



 


