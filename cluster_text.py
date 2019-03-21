import random
import jieba
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import gensim
from gensim.models import Word2Vec
from sklearn.preprocessing import scale
import multiprocessing

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
def preprocess_text(content_lines, sentences):
    for line in content_lines:
        try:
            segs = jieba.lcut(line)
            segs = [v for v in segs if not str(v).isdigit()]
            segs = list(filter(lambda x: x.strip(), segs))
            segs = list(filter(lambda x: len(x)>1, segs))
            segs = list(filter(lambda x: x not in stopwords, segs))
            sentences.append((" ".join(segs)))
        except Exception:
            print(line)
            continue

sentences = []
preprocess_text(laogong, sentences)
preprocess_text(laopo, sentences)
preprocess_text(erzi, sentences)
preprocess_text(nver, sentences)

random.shuffle(sentences)
for sentence in sentences[:10]:
        print(sentence)

# 抽取词向量特征

# 将文本中的词语转换为词频矩阵，矩阵元素a[i][j]表示j词在i类文本下的词频
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
# 统计每个词语的tf-idf权值
transformer = TfidfTransformer()
# 第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
tfidf = transformer.fit_transform(vectorizer.fit_transform(sentences))
# 获取词袋模型中的所有词语
word = vectorizer.get_feature_names()
# 将tf-idf矩阵抽取出来，元素w[i][j]表示j词在i类文本中的tf-idf权重
weight = tfidf.toarray()
# 查看特征大小
print('Feature length: ' + str(len(word)))

# tf-idf k-means聚类
numClass = 4
clf = KMeans(n_clusters=numClass, max_iter=10000, init="k-means++")
pca = PCA(n_components=10)
TnewData = pca.fit_transform(weight)
s = clf.fit(TnewData)
# 聚类结果可视化
def plot_cluster(result, newData, numClass):
    plt.figure(2)
    Lab = [[] for i in range(numClass)]
    index = 0
    for labi in result:
        Lab[labi].append(index)
        index += 1
    color = ['oy', 'ob', 'og', 'cs', 'ms', 'bs', 'ks', 'ys', 'yv', 'mv', 'bv', 'kv', 'gv', 'y^', 'm^', 'b^', 'k^','g^'] * 3
    for i in range(numClass):
        x1 = []
        y1 = []
        for ind1 in newData[Lab[i]]:
            try:
                y1.append(ind1[1])
                x1.append(ind1[0])
            except:
                pass
            plt.plot(x1, y1, color[i])

    # 绘制初始中心点
    x1 = []
    y1 = []
    for ind1 in clf.cluster_centers_:
        try:
            y1.append(ind1[1])
            x1.append(ind1[0])
        except:
            pass
    plt.plot(x1, y1, "rv")
    plt.show()

# 对数据降维到2维，然后获得结果，最后绘制聚类结果图
pca = PCA(n_components=2)
newData = pca.fit_transform(weight)
result = list(clf.predict(TnewData))
plot_cluster(result, newData, numClass)
