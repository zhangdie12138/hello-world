# -*- coding: utf-8 -*-

from glob import glob
import itertools
import os.path
import re
import tarfile
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib import rcParams


from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from Model_Trainer import *
from preprocessing_data import *
import time
from word_vector import *
class Partial_fit:
    """docstring for partial_fit"""

    def __init__(self, training_data,training_target):
        self.training_data=training_data
        self.training_target=training_target

    def progress(self, cls_name, stats,test_stats):
   
        duration = time.time() - stats['t0']
        s = "%20s classifier : \t" % cls_name
        s += "%(n_train)6d train lines (%(n_train_pos)6d positive) " % stats
        s += "%(n_test)6d test lines (%(n_test_pos)6d positive) " % test_stats
        s += "accuracy: %(accuracy).3f " % stats
        s += "in %.2fs (%5d lines/s)" % (duration, stats['n_train'] / duration)
        return s

    def partial_fit(self,test_data,test_target,results_file):
        vectorizer = HashingVectorizer(decode_error='ignore', n_features=2 ** 18,
                               alternate_sign=False)
        cls_stats = {}        
        for cls_name in partial_fit_classifiers:
            stats = {'n_train': 0, 'n_train_pos': 0,
                'accuracy': 0.0, 'accuracy_history': [(0, 0)], 't0': time.time(),
                'runtime_history': [(0, 0)], 'total_fit_time': 0.0}
            cls_stats[cls_name] = stats
        
        stream = stream_data(self.training_data,self.training_target)


        
        test_stream = stream_data(test_data,test_target)
        minibatch_iterators = iter_minibatches(stream,minibatch_size= 10000)
        all_classes = np.array([0, 1])
        tick = time.time()
        X_test_text, y_test = get_minibatch(test_stream, 100000)
        parsing_time = time.time() - tick
        tick = time.time()
        X_test = vectorizer.transform(X_test_text)
        vectorizing_time = time.time() - tick
        # test data statistics
        test_stats = {'n_test': 0, 'n_test_pos': 0}
        test_stats['n_test'] += len(y_test)
        test_stats['n_test_pos'] += y_test.count(0)

        f = open(results_file , 'w',encoding='utf-8')
        f.write("Test set is %d lines (%d positive)" % (len(y_test), y_test.count(0)) + "\n")

        total_vect_time = 0.0

        for i, (X_train_text, y_train) in enumerate(minibatch_iterators):
            tick = time.time()
            X_train = vectorizer.transform(X_train_text)
            total_vect_time += time.time() - tick

            for cls_name, cls in partial_fit_classifiers.items():
                tick = time.time()
              # update estimator with examples in the current mini-bat
                cls.partial_fit(X_train,y_train,classes=all_classes)
                # accumulate test accuracy stats
                # accumulate test accuracy stats
                cls_stats[cls_name]['total_fit_time'] += time.time() - tick
                cls_stats[cls_name]['n_train'] += X_train.shape[0]
                cls_stats[cls_name]['n_train_pos'] += y_train.count(0)
                tick = time.time()
                cls_stats[cls_name]['accuracy'] = cls.score(X_test, y_test)
                cls_stats[cls_name]['prediction_time'] = time.time() - tick
                acc_history = (cls_stats[cls_name]['accuracy'],cls_stats[cls_name]['n_train'])
                cls_stats[cls_name]['accuracy_history'].append(acc_history)
                run_history = (cls_stats[cls_name]['accuracy'],total_vect_time + cls_stats[cls_name]['total_fit_time'])
                cls_stats[cls_name]['runtime_history'].append(run_history)
                if i % 2 == 0:
                    f.write(self.progress(cls_name, cls_stats[cls_name],test_stats)+'\n')
                    				
            if i % 2 == 0:
                f.write('\n')
                f.write('\n')
        f.close()
      
        return cls_stats,total_vect_time,parsing_time,vectorizing_time

        
    def plot_accuracy(self,x, y, x_legend):

        x = np.array(x)
        y = np.array(y)
        plt.title('Classification accuracy as a function of %s' % x_legend)
        plt.xlabel('%s' % x_legend)
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.plot(x, y)

    def autolabel(self,ax,rectangles):
  
        for rect in rectangles:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2.,
                    1.05 * height, '%.4f' % height,
                    ha='center', va='bottom')
    def plot(self,cls_stats,total_vect_time,parsing_time,vectorizing_time):
        rcParams['legend.fontsize'] = 10
        cls_names = list(sorted(cls_stats.keys()))
# Plot accuracy evolution
        plt.figure()
        for _, stats in sorted(cls_stats.items()):
    # Plot accuracy evolution with #examples
            accuracy, n_examples = zip(*stats['accuracy_history'])
            self.plot_accuracy(n_examples, accuracy, "training examples (#)")
            ax = plt.gca()
            ax.set_ylim((0.8, 1))
        plt.legend(cls_names, loc='best')

        plt.figure()
        for _, stats in sorted(cls_stats.items()):
    # Plot accuracy evolution with runtime
            accuracy, runtime = zip(*stats['runtime_history'])
            self.plot_accuracy(runtime, accuracy, 'runtime (s)')
            ax = plt.gca()
            ax.set_ylim((0.8, 1))
        plt.legend(cls_names, loc='best')

# Plot fitting times
        plt.figure()
        fig = plt.gcf()
        cls_runtime = []
        for cls_name, stats in sorted(cls_stats.items()):
            cls_runtime.append(stats['total_fit_time'])

        cls_runtime.append(total_vect_time)
        cls_names.append('Vectorization')
        
       
        bar_colors = ['b', 'g', 'r', 'c', 'm', 'y']

        ax = plt.subplot(111)
        rectangles = plt.bar(range(len(cls_names)), cls_runtime, width=0.5,
                        color=bar_colors)

        ax.set_xticks(np.linspace(0.25, len(cls_names) - 0.75, len(cls_names)))
        ax.set_xticklabels(cls_names, fontsize=10)
        ymax = max(cls_runtime) * 1.2
        ax.set_ylim((0, ymax))
        ax.set_ylabel('runtime (s)')
        ax.set_title('Training Times')
        self.autolabel(ax,rectangles)
        plt.show()

# Plot prediction times
        plt.figure()
        cls_runtime = []
        cls_names = list(sorted(cls_stats.keys()))
        for cls_name, stats in sorted(cls_stats.items()):
            cls_runtime.append(stats['prediction_time'])
        cls_runtime.append(parsing_time)
        cls_names.append('Read/Parse\n+Feat.Extr.')
        cls_runtime.append(vectorizing_time)
        cls_names.append('Hash\n+Vect.')
        
        
        ax = plt.subplot(111)
        rectangles = plt.bar(range(len(cls_names)), cls_runtime, width=0.5,
                        color=bar_colors)

        ax.set_xticks(np.linspace(0.25, len(cls_names) - 0.75, len(cls_names)))
        ax.set_xticklabels(cls_names, fontsize=8)
        plt.setp(plt.xticks()[1], rotation=30)
        ymax = max(cls_runtime) * 1.2
        ax.set_ylim((0, ymax))
        ax.set_ylabel('runtime (s)')
        ax.set_title('Prediction Times (%d instances)' % 1000)
        self.autolabel(ax,rectangles)
        plt.show()



    

    