import random
import jieba
import pandas as pd

# 加载停用词
stopwords = pd.read_csv('stopwords.txt', index_col=False, quoting=3, sep="\t", names=['stopword'], encoding='utf-8')
stopwords = stopwords['stopword'].values

# 加载语料
laogong_df = pd.read_csv('beilaogongda.csv', encoding='utf-8', sep=',')
laopo_df = pd.read_csv('beilaopoda.csv', encoding='utf-8', sep=',')
erzi_df = pd.read_csv('beierzida.csv', encoding='utf-8', sep=',')
nver_df = pd.read_csv('beinverda.csv', encoding='utf-8', sep=',')

# 删除语料的nan行
laogong_df.dropna(inplace=True)
laopo_df.dropna(inplace=True)
erzi_df.dropna(inplace=True)
nver_df.dropna(inplace=True)

# 转换
laogong = laogong_df.segment.values.tolist()
laopo = laopo_df.segment.values.tolist()
erzi = erzi_df.segment.values.tolist()
nver = nver_df.segment.values.tolist()

# 定义分词和打标签函数preprocess_text
# 参数content_liness即为上面转换的list
# 参数sentences是定义的空list，用来存储打标签之后的数据
# 参数category是类型标签
def preprocess_text(content_lines, sentences, category):
    for line in content_lines:
        try:
            segs = jieba.lcut(line)
            segs = [v for v in segs if not str(v).isdigit()]
            segs = list(filter(lambda x: x.strip(), segs))
            segs = list(filter(lambda x: len(x)>1, segs))
            segs = list(filter(lambda x: x not in stopwords, segs))
            sentences.append((" ".join(segs), category))
        except Exception:
            print(line)
            continue

sentences = []
preprocess_text(laogong, sentences, 0)
preprocess_text(laopo, sentences, 1)
preprocess_text(erzi, sentences, 2)
preprocess_text(nver, sentences, 3)

random.shuffle(sentences)

for sentence in sentences[:10]:
    print(sentence[0], sentence[1])

# 抽取词向量特征
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer(
    analyzer='word',
    max_features=4000,
)

from sklearn.model_selection import train_test_split
x, y = zip(*sentences)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1256)

vec.fit(x_train)

# 算法建模和模型训练

# 训练模型：NB
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(vec.transform(x_train), y_train)

print(classifier.score(vec.transform(x_test), y_test))

vec2 = CountVectorizer(
    analyzer='word',
    ngram_range=(1,4),
    max_features=20000,
)
vec2.fit(x_train)

classifier2 = MultinomialNB()
classifier2.fit(vec2.transform(x_train), y_train)
print(classifier2.score(vec2.transform(x_test), y_test))

# 训练模型：SVM
from sklearn.svm import SVC
svm = SVC(kernel='linear')
svm.fit(vec.transform(x_train), y_train)
print(svm.score(vec.transform(x_test), y_test))

import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
import numpy as np
# xgb矩阵赋值
xgb_train = xgb.DMatrix(vec.transform(x_train), label=y_train)
xgb_test = xgb.DMatrix(vec.transform(x_test))

params = {
            'booster': 'gbtree',     #使用gbtree
            'objective': 'multi:softmax',  # 多分类的问题、
            # 'objective': 'multi:softprob',   # 多分类概率
            #'objective': 'binary:logistic',  #二分类
            'eval_metric': 'merror',   #logloss
            'num_class': 4,  # 类别数，与 multisoftmax 并用
            'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
            'max_depth': 8,  # 构建树的深度，越大越容易过拟合
            'alpha': 0,   # L1正则化系数
            'lambda': 10,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
            'subsample': 0.7,  # 随机采样训练样本
            'colsample_bytree': 0.5,  # 生成树时进行的列采样
            'min_child_weight': 3,
            # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
            # 假设 h 在 0.01 附近，min_child_weight 为 1 叶子节点中最少需要包含 100 个样本。
            'silent': 0,  # 设置成1则没有运行信息输出，最好是设置为0.
            'eta': 0.03,  # 如同学习率
            'seed': 1000,
            'nthread': -1,  # cpu 线程数
            'missing': 1
}
num_round = 4
bst = xgb.train(params,xgb_train,num_round)
bst.save_model('xgboost.model')

preds = bst.predict(xgb_test) # 得到的是第一类别的概率
p_label = [round(value) for value in preds] # 得到预测标签
