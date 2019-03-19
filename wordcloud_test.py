# wordcloud制作词云


# coding: utf-8

# In[4]:


import jieba
import pandas as pd
import numpy as np
from scipy.misc import imread
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt


# In[ ]:


# 定义文件路径
dir = "C://Users//qufang//NLP_zh_learning//"

# 定义语料文件路径
file = "".join([dir, "z_m.csv"])

# 定义停用词文件路径
stop_words = "".join([dir, "stopwords.txt"])

# 定义wordcloud中字体文件的路径
simhei = "".join([dir, "simhei.ttf"])

# 读取语料
df = pd.read_csv(file, encoding='utf-8')
df.head()

# 如果存在nan，删除
df.dropna(inplace=True)

# 将content一列转为list
content = df.content.values.tolist()

# 用jieba进行分词操作
segment = []
for line in content:
    try:
        segs = jieba.cut_for_search(line)
        segs = [v for v in segs if not str(v).isdigit()] #去数字
        segs = list(filter(lambda x: x.strip(), segs)) # 去左右空格
        for seg in segs:
            if len(seg) > 1 and seg != '\r\n':
                segment.append(seg)
    except:
        print(line)
        continue

# 分词后加入一个新的DataFrame
word_df = pd.DataFrame({'segment': segment})

# 加载停用词
stopwords = pd.read_csv(stop_words, index_col=False, quoting=3, sep="\t", names=['stopwords'], encoding='gbk')

# 安装关键字groupby分组统计词频，并按照降序排序
words_stat = word_df.groupby(by=['segment'])['segment'].agg({"计数": np.size})
words_stat = words_stat.reset_index().sort_values(by=['计数'], ascending=False)

# 分词之后去掉停用词
words_stat = words_stat[~words_stat.segment.isin(stopwords.stopwords)]

# 下面是重点，绘制wordcloud词云，提供两种方式
# 第一种是默认的样式
wordcloud = WordCloud(font_path=simhei, background_color="white", max_font_size=80)
word_frequence = {x[0]:x[1] for x in words_stat.head(1000).values}
wordcloud = wordcloud.fit_words(word_frequence)
plt.imshow(wordcloud)
wordcloud.to_file(r'wordcloud1.jpg') #保存结果

# 第二种是自定义图片
text = " ".join(words_stat['segment'].head(100).astype(str))
abel_mask = imread(r"china.jpg") #这里设置了一张中国地图
wordcloud2 = WordCloud(background_color='white', #设置背景颜色
                      mask =abel_mask, #设置背景图片
                      max_words=3000, #设置最大显示的字数
                      font_path=simhei, #设置字体格式
                      width=2048,
                      height=1024,
                      scale=4.0,
                      max_font_size=300, #字体最大值
                      random_state=42).generate(text)

# 根据图片生成词云颜色
image_colors = ImageColorGenerator(abel_mask)
wordcloud2.recolor(color_func=image_colors)

# 以下代码显示图片
plt.imshow(wordcloud2)
plt.axis("off")
plt.show()
wordcloud2.to_file(r'wordcloud2.jpg') #保存结果


