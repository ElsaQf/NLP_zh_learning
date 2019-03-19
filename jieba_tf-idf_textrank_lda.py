# 提取中文关键词：jieba: tf-idf textrank lda


import jieba.analyse as analyse
import jieba
import pandas as pd
from gensim import corpora, models, similarities
import gensim
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

# 设置文件路径
dir = "C://Users//qufang//NLP_zh_learning//"
file_desc = "".join([dir, 'car.csv'])
stop_words = "".join([dir, 'stopwords.txt'])

# 定义停用词
stopwords = pd.read_csv(stop_words, index_col=False, quoting=3, sep="\t", names=['stopword'], encoding='gbk')
stopwords = stopwords['stopword'].values

# 加载语料
df = pd.read_csv(file_desc, encoding='utf-8')

# 删除nan行
df.dropna(inplace=True)
lines = df.content.values.tolist()

# 开始分词
sentences = []
for line in lines:
    try:
        segs = jieba.lcut(line)
        segs = [v for v in segs if not str(v).isdigit()] #去数字
        segs = list(filter(lambda x: x.strip(), segs)) #去左右空格
        segs = list(filter(lambda x: x not in stopwords, segs)) #去停用词
        sentences.append(segs)
    except Exception:
        print(line)
        continue
# 构建词袋模型
dictionary = corpora.Dictionary(sentences)
corpus = [dictionary.doc2bow(sentence) for sentence in sentences]

# lda模型，num_topics是主题的个数，这里定义了5个
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=10)

# 我们查一下第1号分类，其中最常出现的5个词是：
print(lda.print_topic(1, topn=5))

# 我们打印所有5个主题，每个主题显示8个词
for topic in lda.print_topics(num_topics=10, num_words=8):
    print(topic[1])
