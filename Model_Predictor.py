# -*- coding: utf-8 -*-
from sklearn import metrics
import time

class Predictor:
    def __init__(self,clf):

        self.clf = clf

    def sample_predict(self,data,data_target):
        self.data_target = data_target
        self.data=data
        test_result = self.clf.predict(self.data)
        return metrics.classification_report(self.data_target, test_result), metrics.confusion_matrix(self.data_target, test_result)

    def new_predict(self,data,fr):
        self.data=data
        tick = time.time()
        predict_result = self.clf.predict(self.data)
        time = time.time()-tick    
        for i in range(len(predict_result)):
            fr.write(predict_result[i])
        fr.write('time'+'\n')
        


