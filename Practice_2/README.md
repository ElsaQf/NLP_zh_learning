## 训练赛——O2O商铺食品安全相关评论发现
### 赛题任务
    本赛题提供了10000条对O2O店铺的评论文本训练数据，分为与食品安全有关和与食品安全无关两个类别。
    参赛者需要根据训练集构造文本分类模型，预测2000条测试集中的评论是否与食品安全有关。
    
#### 方法
1. CountVectorizer / TfidfVecotrizer
2. MultimonalNB / SVM(kernel='linear')
3. StratifiedKFold
4. CNN / LSTM
5. 主题提取（关键词k1,k2, ... -> 命中 -> “食品安全”词库）
