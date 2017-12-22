# -*- coding: utf-8 -*-
import numpy as np
import jieba
import sklearn.feature_extraction.text
from scipy import sparse, io
import re
import load_data
from sklearn import feature_extraction  
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer

def fenci(content_file,fenci_file,stopwords):
    ff = open(fenci_file,'w+',encoding='utf-8')
    with open(content_file, 'r',encoding='utf-8') as f:
        a=f.readlines()
        for line in a:
            words= jieba.cut(line)
            final = ''
            for seg in words:  
                if seg not in stopwords:
                    final += seg              
            segs=jieba.cut(final) 
            for se in segs:      
                if (se != ' ' and se != "\n" and se != "\n\n"):
                    ff.write(se+"\t")
                else:
                    continue
            ff.write("\n")
    ff.close()
    
    
def tf_idf(fenci_file,word_vector_file):
    with open(fenci_file,'r',encoding='utf-8') as f:
        lines=[]
        for a in f.readlines():
            line=a.rstrip("\n")
            lines.append(line)
        vectorizer=CountVectorizer()               
        transformer=TfidfTransformer()
        tfidf=transformer.fit_transform(vectorizer.fit_transform(lines))
        tfidf=tfidf.astype(np.float32)
        word=vectorizer.get_feature_names()
        io.mmwrite(word_vector_file, tfidf)
    
def loadStopWords(filepath):
    result=[]
    stopwordfile = open(filepath,"r",encoding="utf-8").read().splitlines()
    for line in stopwordfile:
        result.append(line)
    stopwords={}.fromkeys(result)
    return stopwords

