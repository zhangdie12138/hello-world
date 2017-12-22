# Spam_messages
垃圾短信分类

## 任务目标
实现一个垃圾短信识别系统，在给定的数据集上验证效果

## 数据介绍
带标签数据80W条，其中垃圾短信(label=1)8W条，正常短信(label=0)72W条
不带标签数据20W行(有部分空行)

## 选取数据说明
实验随机选取（10%）正例(label=0)72000条和负例(label=1)8000条进行。

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
   * 将80万条随机选取8000条进行增量训练实验，得到四种训练器下的分类精度和所需时间等图








