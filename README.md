# NLP_zh_learning
中文自然语言处理入门课程代码实现

![中文自然语言处理流程](https://github.com/ElsaQf/NLP_zh_learning/blob/master/routine.JPG)

### 步骤
    1.读取数据，数据格式转换(分词，去停用词，shuffle)
    2.x_train, y_train, x_test, y_test = train_test_split()
    3.vec.fit(x_train)
    4.classifier.fit(vec.transform(x_train), y_train)
    5.classifier.predict(vec.transform(x_test))
    6.classifier.score(vec.transform(x_test), y_test)
