from time import time
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from scipy import sparse, io
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
import numpy as np
import itertools

# 利用生成器迭代
def stream_data(content,label):
  for i in range(len(content)):
    yield content[i],label[i]

def get_minibatch(stream,size):
  X_train, y_train = [],[]
  try:
    for i in range(size):
      temp = next(stream)
      X_train.append(temp[0])
      y_train.append(int(temp[1]))
  except StopIteration:
    pass
  return X_train,y_train


def iter_minibatches(stream,minibatch_size=1000):
  X_train,y_train = get_minibatch(stream,minibatch_size)
  while (len(X_train)):
    yield X_train,y_train
    X_train,y_train = get_minibatch(stream,minibatch_size)

def dimensionality_reduction(training_data, test_data, type='pca'):
    if type == 'pca':
        n_components = 1000
        t0 = time()
        pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True)
        pca.fit(training_data)
        print("done in %0.3fs" % (time() - t0))
        t0 = time()
        training_data_transform = sparse.csr_matrix(pca.transform(training_data))
        test_data_transform = sparse.csr_matrix(pca.transform(test_data))
        print("done in %0.3fs" % (time() - t0))
        #random_projections
        #feature_agglomeration
        return training_data_transform, test_data_transform

#svd数据降维
"""
def dimensionality_reduction(data, type='svd'):
  if type == 'svd':
    n_components = 1000
    t0 = time()
    #svd = TruncatedSVD(n_components= n_components, random_state=42)
    svd=TruncatedSVD(n_components=n_components, algorithm='randomized', random_state=None, n_iter=5, tol=0.0)
    #pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True)
    svd.fit(data) 
    print("done in %0.3fs" % (time() - t0))
    t0 = time()
    data_transform = sparse.csr_matrix(svd.transform(data))
    print("done in %0.3fs" % (time() - t0))
    #random_projections
    #feature_agglomeration
    return data_transform
    """

def split_data(content, label):
  training_data, test_data, training_target, test_target = train_test_split(content, label, test_size=0.2, random_state=20)
  return training_data, test_data, training_target, test_target
