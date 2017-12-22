# -*- coding: utf-8 -*-
import numpy as np
from sklearn import svm
from sklearn import metrics
from scipy import sparse, io

from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from scipy import sparse, io
from sklearn.decomposition import PCA
from preprocessing_data import split_data
from preprocessing_data import dimensionality_reduction
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model.logistic import LogisticRegression




#增量训练的分类器
partial_fit_classifiers = {
    'SGD': SGDClassifier(),
    'Perceptron': Perceptron(),
    'NB Multinomial': MultinomialNB(alpha=0.01),
    'Passive-Aggressive': PassiveAggressiveClassifier(),
}



class SVMLinear:
    def __init__(self, training_data, training_target):
        self.training_data = training_data
        self.training_target = training_target
        self.clf = svm.SVC(C=1, class_weight=None, coef0=0.0,
                           decision_function_shape=None, degree=3, gamma='auto',
                           kernel='linear', max_iter=-1, probability=False,
                           random_state=None, shrinking=True, tol=0.001, verbose=False)

    def learn_best_param(self):
        C_range = np.logspace(-2, 10, 13)
        param_grid = dict(C=C_range)
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
        grid.fit(self.training_data, self.training_target)
        self.clf.set_params(C=grid.best_params_['C'])
        print("The best parameters are %s with a score of %0.2f"
              % (grid.best_params_, grid.best_score_))

    def train_classifier(self,model_file):
        self.learn_best_param()
        self.clf.fit(self.training_data, self.training_target)
        joblib.dump(self.clf, model_file)
        training_result = self.clf.predict(self.training_data)
        print (metrics.classification_report(self.training_target, training_result))

    def cross_validation(self):
        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=20)
        scores = cross_val_score(self.clf, self.training_data, self.training_target, cv=cv, scoring='f1_macro')
        print (scores)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


class LogisticReg:
    """docstring for logistic"""
    def __init__(self, training_data, training_target):
        self.training_data = training_data
        self.training_target = training_target
        self.clf = GaussianNB()
class Trainer_bayes:
    def __init__(self, training_data, training_target):
        self.training_data = training_data
        self.training_target = training_target
        self.clf = GaussianNB()


    def train_classifier(self,model_file):
        self.clf.fit(self.training_data, self.training_target)
        joblib.dump(self.clf, model_file)
        training_result = self.clf.predict(self.training_data)
        print (metrics.classification_report(self.training_target, training_result))

    def cross_validation(self):
        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=20)
        scores = cross_val_score(self.clf, self.training_data, self.training_target, cv=cv, scoring='f1_macro')
        print (scores)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



class SVMRbf:
    def __init__(self, training_data, training_target):
        self.training_data = training_data
        self.training_target = training_target
        self.clf = svm.SVC(C=100, class_weight=None, coef0=0.0,
                           decision_function_shape=None, degree=3, gamma=0.01,
                           kernel='rbf', max_iter=-1, probability=False,
                           random_state=None, shrinking=True, tol=0.001, verbose=False)

    def learn_best_param(self):
        C_range = np.logspace(-2, 10, 13)
        gamma_range = np.logspace(-9, 3, 13)
        param_grid = dict(gamma=gamma_range, C=C_range)
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
        grid.fit(self.training_data, self.training_target)
        self.clf.set_params(C=grid.best_params_['C'], gamma=grid.best_params_['gamma'])
        print("The best parameters are %s with a score of %0.2f"
              % (grid.best_params_, grid.best_score_))
        

    def draw_visualization_param_effect(self, grid, C_range, gamma_range):
        scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),
                                                             len(gamma_range))
        plt.figure(figsize=(8, 6))
        plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
        plt.imshow(scores, interpolation='nearest',
                   norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
        plt.xlabel('gamma')
        plt.ylabel('C')
        plt.colorbar()
        plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
        plt.yticks(np.arange(len(C_range)), C_range)
        plt.title('Validation accuracy')
        plt.savefig('fig/param_effect.png')
        plt.show()

    def train_classifier(self,model_file):
        self.learn_best_param()
        self.clf.fit(self.training_data, self.training_target)
        joblib.dump(self.clf, model_file)
        training_result = self.clf.predict(self.training_data)
        print (metrics.classification_report(self.training_target, training_result))








