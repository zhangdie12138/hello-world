# -*- coding: utf-8 -*-
from numpy import *
import random


# 加载原始数据，进行分割
def load_message(label_messages_file):
    lines =[]
    mess_Po=[]
    mess_Ne=[]
    with open(label_messages_file, encoding='utf-8') as fr:
        for line in fr.readlines():
            lines.append(line)
        num = len(lines)
        for i in range(num):
            message = lines[i].split('\t')
            if message[0]== '0':
                mess_Po.append(lines[i])
            else:
                mess_Ne.append(lines[i])
    print (len(mess_Po))
    return num, mess_Po, mess_Ne

def load_samples(sample_file):
    content=[]
    label=[]
    with open(sample_file, encoding='utf-8') as fr:
        for line in fr.readlines():
            message=line.split('\t')
            label.append(message[0])
            content.append(message[1])
    return content,label


#随机选择10%的数据,正例负例
def data_selection(mess_Po,mess_Ne,sample_file,size=10):
    
    po=[]
    ne=[]
    lens = [int(len(mess_Po)/size),int(len(mess_Ne)/size)]

    rans = [range(len(mess_Po)),range(len(mess_Ne))]
    
    seqs = [random.sample(rans[0],lens[0]),random.sample(rans[1],lens[1])]
    for i in seqs[0]:
        po.append(mess_Po[i])
    for j in seqs[1]:
        ne.append(mess_Ne[j])
    with open(sample_file , 'w',encoding='utf-8') as f:
        for a in po:
            f.write(a)
        for b in ne:
            f.write(b)

def data_storage(content_file,label_file,content, label):
    with open(content_file , 'w',encoding='utf-8') as f:
        for i in range(len(content)):
            f.write(content[i])
        
    with open(label_file, 'w',encoding='utf-8') as f:
        for i in range(len(label)):
            f.write(label[i]+"\n")

