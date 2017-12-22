# Spam_messages
垃圾短信分类

## 任务目标
实现一个垃圾短信识别系统，在给定的数据集上验证效果

## 数据介绍
带标签数据80W条，其中垃圾短信(label=1)8W条，正常短信(label=0)72W条
不带标签数据20W行(有部分空行)

****
## 代码说明
* load_data.py
    * 从原始数据label-messages.txt中随机选取80000条实验数据得到sample-messages.txt
    * 将sample-messages.txt数据加载为content.txt和label.txt，将短信内容与标签分离
* word_vector.py
    * 首先利用jieba包，将内容content.txt进行分词，得到content_fenci.txt，其中去除了停用词
        * [停用词来源](https://raw.githubusercontent.com/CuiCh/Spam_Message_Classification)
    * 利用CountVectorizer和TfidfTransformer计算tf-idf值的词向量，得到word_vector.mtx
* preprocessing_data.py
    * train_test_split将样本数据分为训练数据和测试数据
    * SVD和PCA降维
    * 流式数据：实现一个生成器，每调用一次提供一份小batch数据
* Model_Trainer.py
   * 增量训练的分类器：SGDClassifier、Perceptron、MultinomialNB、PassiveAggressiveClassifier
   * 基础的SVM分类器：SVM线性分类器（构造函数最佳参数C的搜索方法采用GridSearchCV的方式）和SVM-RBF分类器（在RBF核下寻找最佳参数也是通过GridSearchCV方式来自动搜索）
   * 在GridSearchCV下得到最佳参数，将持久化后模型保存。
* Model_Predictor.py 
   * 分为测试数据预测（用于评估）和待分类数据的预测
* partial_fit.py
   * 增量训练并保存结果为Results.txt
   * 绘制不同分类器结果图

* Main.py
   * 主程序可以随机选择数据量进行增量实验，得到四种训练器下的分类精度和所需时间等图，结果存放Data里。

****
## 结果展示
|图片|图片|
|---|---
|![](https://github.com/zhangdie12138/hello-world/blob/master/Data/80w/Accuracy_80w.png)|![](https://github.com/zhangdie12138/hello-world/blob/master/Data/80w/Accuracy_time_80w.png)
|![](https://github.com/zhangdie12138/hello-world/blob/master/Data/80w/Predictiontime_80w.png)|![](https://github.com/zhangdie12138/hello-world/blob/master/Data/80w/Trainingtime_80w.png)

## [Results](https://github.com/zhangdie12138/hello-world/blob/master/Data/80w/Results.txt)
      Test set is 100000 lines (89964 positive)
            Passive-Aggressive classifier : 	 20000 train lines ( 18007 positive) 100000 test lines ( 89964 positive) accuracy: 0.981 in 1.36s (14677 lines/s)
            SGD classifier : 	 20000 train lines ( 18007 positive) 100000 test lines ( 89964 positive) accuracy: 0.975 in 1.41s (14186 lines/s)
            NB Multinomial classifier : 	 20000 train lines ( 18007 positive) 100000 test lines ( 89964 positive) accuracy: 0.974 in 1.47s (13634 lines/s)
            Perceptron classifier : 	 20000 train lines ( 18007 positive) 100000 test lines ( 89964 positive) accuracy: 0.968 in 1.50s (13293 lines/s)

            Passive-Aggressive classifier : 	 60000 train lines ( 54081 positive) 100000 test lines ( 89964 positive) accuracy: 0.985 in 2.15s (27923 lines/s)
            SGD classifier : 	 60000 train lines ( 54081 positive) 100000 test lines ( 89964 positive) accuracy: 0.975 in 2.19s (27443 lines/s)
            NB Multinomial classifier : 	 60000 train lines ( 54081 positive) 100000 test lines ( 89964 positive) accuracy: 0.979 in 2.24s (26732 lines/s)
            Perceptron classifier : 	 60000 train lines ( 54081 positive) 100000 test lines ( 89964 positive) accuracy: 0.978 in 2.28s (26286 lines/s)
            
            Passive-Aggressive classifier : 	100000 train lines ( 90145 positive) 100000 test lines ( 89964 positive) accuracy: 0.987 in 2.95s (33946 lines/s)
            SGD classifier : 	100000 train lines ( 90145 positive) 100000 test lines ( 89964 positive) accuracy: 0.976 in 2.98s (33506 lines/s)
            NB Multinomial classifier : 	100000 train lines ( 90145 positive) 100000 test lines ( 89964 positive) accuracy: 0.981 in 3.05s (32818 lines/s)
            Perceptron classifier : 	100000 train lines ( 90145 positive) 100000 test lines ( 89964 positive) accuracy: 0.980 in 3.09s (32402 lines/s)
            
            Passive-Aggressive classifier : 	140000 train lines (126172 positive) 100000 test lines ( 89964 positive) accuracy: 0.988 in 3.79s (36977 lines/s)
            SGD classifier : 	140000 train lines (126172 positive) 100000 test lines ( 89964 positive) accuracy: 0.975 in 3.83s (36556 lines/s)
            NB Multinomial classifier : 	140000 train lines (126172 positive) 100000 test lines ( 89964 positive) accuracy: 0.982 in 3.89s (35944 lines/s)
            Perceptron classifier : 	140000 train lines (126172 positive) 100000 test lines ( 89964 positive) accuracy: 0.982 in 3.94s (35510 lines/s)
            Passive-Aggressive classifier : 	180000 train lines (162172 positive) 100000 test lines ( 89964 positive) accuracy: 0.988 in 4.60s (39145 lines/s)
            SGD classifier : 	180000 train lines (162172 positive) 100000 test lines ( 89964 positive) accuracy: 0.975 in 4.64s (38823 lines/s)
            NB Multinomial classifier : 	180000 train lines (162172 positive) 100000 test lines ( 89964 positive) accuracy: 0.983 in 4.70s (38293 lines/s)
            Perceptron classifier : 	180000 train lines (162172 positive) 100000 test lines ( 89964 positive) accuracy: 0.983 in 4.74s (37985 lines/s)
            Passive-Aggressive classifier : 	220000 train lines (198212 positive) 100000 test lines ( 89964 positive) accuracy: 0.989 in 5.45s (40381 lines/s)
            SGD classifier : 	220000 train lines (198212 positive) 100000 test lines ( 89964 positive) accuracy: 0.976 in 5.50s (40031 lines/s)
            NB Multinomial classifier : 	220000 train lines (198212 positive) 100000 test lines ( 89964 positive) accuracy: 0.984 in 5.56s (39544 lines/s)
            Perceptron classifier : 	220000 train lines (198212 positive) 100000 test lines ( 89964 positive) accuracy: 0.984 in 5.61s (39230 lines/s)
            
            Passive-Aggressive classifier : 	260000 train lines (234181 positive) 100000 test lines ( 89964 positive) accuracy: 0.989 in 6.28s (41416 lines/s)
            SGD classifier : 	260000 train lines (234181 positive) 100000 test lines ( 89964 positive) accuracy: 0.976 in 6.32s (41137 lines/s)
            NB Multinomial classifier : 	260000 train lines (234181 positive) 100000 test lines ( 89964 positive) accuracy: 0.984 in 6.39s (40714 lines/s)
            Perceptron classifier : 	260000 train lines (234181 positive) 100000 test lines ( 89964 positive) accuracy: 0.984 in 6.43s (40460 lines/s)
            
            Passive-Aggressive classifier : 	300000 train lines (270226 positive) 100000 test lines ( 89964 positive) accuracy: 0.989 in 7.10s (42245 lines/s)
            SGD classifier : 	300000 train lines (270226 positive) 100000 test lines ( 89964 positive) accuracy: 0.976 in 7.14s (42010 lines/s)
            NB Multinomial classifier : 	300000 train lines (270226 positive) 100000 test lines ( 89964 positive) accuracy: 0.984 in 7.20s (41674 lines/s)
            Perceptron classifier : 	300000 train lines (270226 positive) 100000 test lines ( 89964 positive) accuracy: 0.984 in 7.24s (41455 lines/s)
            
            Passive-Aggressive classifier : 	340000 train lines (306224 positive) 100000 test lines ( 89964 positive) accuracy: 0.990 in 7.91s (42986 lines/s)
            SGD classifier : 	340000 train lines (306224 positive) 100000 test lines ( 89964 positive) accuracy: 0.976 in 7.95s (42747 lines/s)
            NB Multinomial classifier : 	340000 train lines (306224 positive) 100000 test lines ( 89964 positive) accuracy: 0.985 in 8.02s (42379 lines/s)
            Perceptron classifier : 	340000 train lines (306224 positive) 100000 test lines ( 89964 positive) accuracy: 0.984 in 8.07s (42142 lines/s)
          
            Passive-Aggressive classifier : 	380000 train lines (342079 positive) 100000 test lines ( 89964 positive) accuracy: 0.990 in 8.96s (42409 lines/s)
            SGD classifier : 	380000 train lines (342079 positive) 100000 test lines ( 89964 positive) accuracy: 0.976 in 9.00s (42227 lines/s)
            NB Multinomial classifier : 	380000 train lines (342079 positive) 100000 test lines ( 89964 positive) accuracy: 0.985 in 9.06s (41949 lines/s)
            Perceptron classifier : 	380000 train lines (342079 positive) 100000 test lines ( 89964 positive) accuracy: 0.986 in 9.10s (41741 lines/s)
            
            Passive-Aggressive classifier : 	420000 train lines (378088 positive) 100000 test lines ( 89964 positive) accuracy: 0.990 in 9.78s (42938 lines/s)
            SGD classifier : 	420000 train lines (378088 positive) 100000 test lines ( 89964 positive) accuracy: 0.976 in 9.84s (42695 lines/s)
            NB Multinomial classifier : 	420000 train lines (378088 positive) 100000 test lines ( 89964 positive) accuracy: 0.985 in 9.91s (42386 lines/s)
            Perceptron classifier : 	420000 train lines (378088 positive) 100000 test lines ( 89964 positive) accuracy: 0.985 in 9.95s (42211 lines/s)
            
            Passive-Aggressive classifier : 	460000 train lines (414081 positive) 100000 test lines ( 89964 positive) accuracy: 0.990 in 10.69s (43045 lines/s)
            SGD classifier : 	460000 train lines (414081 positive) 100000 test lines ( 89964 positive) accuracy: 0.976 in 10.72s (42898 lines/s)
            NB Multinomial classifier : 	460000 train lines (414081 positive) 100000 test lines ( 89964 positive) accuracy: 0.985 in 10.79s (42635 lines/s)
            Perceptron classifier : 	460000 train lines (414081 positive) 100000 test lines ( 89964 positive) accuracy: 0.987 in 10.83s (42469 lines/s)
            
            Passive-Aggressive classifier : 	500000 train lines (450129 positive) 100000 test lines ( 89964 positive) accuracy: 0.991 in 11.48s (43540 lines/s)
            SGD classifier : 	500000 train lines (450129 positive) 100000 test lines ( 89964 positive) accuracy: 0.975 in 11.52s (43394 lines/s)
            NB Multinomial classifier : 	500000 train lines (450129 positive) 100000 test lines ( 89964 positive) accuracy: 0.986 in 11.58s (43178 lines/s)
            Perceptron classifier : 	500000 train lines (450129 positive) 100000 test lines ( 89964 positive) accuracy: 0.986 in 11.62s (43022 lines/s)
            
            Passive-Aggressive classifier : 	540000 train lines (486119 positive) 100000 test lines ( 89964 positive) accuracy: 0.991 in 12.28s (43962 lines/s)
            SGD classifier : 	540000 train lines (486119 positive) 100000 test lines ( 89964 positive) accuracy: 0.976 in 12.32s (43821 lines/s)
            NB Multinomial classifier : 	540000 train lines (486119 positive) 100000 test lines ( 89964 positive) accuracy: 0.986 in 12.38s (43604 lines/s)
            Perceptron classifier : 	540000 train lines (486119 positive) 100000 test lines ( 89964 positive) accuracy: 0.986 in 12.43s (43460 lines/s)
            
            Passive-Aggressive classifier : 	580000 train lines (522097 positive) 100000 test lines ( 89964 positive) accuracy: 0.991 in 13.10s (44280 lines/s)
            SGD classifier : 	580000 train lines (522097 positive) 100000 test lines ( 89964 positive) accuracy: 0.976 in 13.14s (44140 lines/s)
            NB Multinomial classifier : 	580000 train lines (522097 positive) 100000 test lines ( 89964 positive) accuracy: 0.986 in 13.21s (43900 lines/s)
            Perceptron classifier : 	580000 train lines (522097 positive) 100000 test lines ( 89964 positive) accuracy: 0.986 in 13.25s (43774 lines/s)
            
            Passive-Aggressive classifier : 	620000 train lines (558018 positive) 100000 test lines ( 89964 positive) accuracy: 0.991 in 13.98s (44353 lines/s)
            SGD classifier : 	620000 train lines (558018 positive) 100000 test lines ( 89964 positive) accuracy: 0.976 in 14.03s (44202 lines/s)
            NB Multinomial classifier : 	620000 train lines (558018 positive) 100000 test lines ( 89964 positive) accuracy: 0.986 in 14.09s (43999 lines/s)
            Perceptron classifier : 	620000 train lines (558018 positive) 100000 test lines ( 89964 positive) accuracy: 0.985 in 14.13s (43865 lines/s)
