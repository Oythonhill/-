
>Created by yinhongyu at 2018-4-28

>email: hyhyin@163.com

>使用jieba和sklearn实现了tf idf的计算


```python
import jieba
import jieba.posseg as pseg
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import re
```

### 1 读取数据文件
>数据爬取自新浪新闻，以"中美贸易战"为关键词，按照相关度搜索，爬取了搜索结果的前100页新闻的正文；


```python
# 读取数据文件
sina_news = pd.read_excel(r"C:\Users\YHY\Desktop\sina_news_finally.xlsx")
sina_news.head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>标题</th>
      <th>来源</th>
      <th>内容</th>
      <th>时间</th>
      <th>阶段</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>外交部回应"美对华贸易调查":打贸易战只会双输</td>
      <td>海外网</td>
      <td>海外网8月14日电在14日的外交部例行记者会上，发言人华春莹就近日热点进行回应。相关内容如下...</td>
      <td>2017-08-14</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>特朗普政府对华 “301条款战”一触即发，中美贸易战只会双输</td>
      <td>一财网</td>
      <td>针对美国总统特朗普将签署行政备忘录，对中国发起贸易调查一事，中国外交部发言人华春莹14日回应...</td>
      <td>2017-08-14</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>特朗普欲对华发起301条款调查 专家：该做法已过时</td>
      <td>第一财经日报</td>
      <td>特朗普欲对华动用“301条款”被指“过时了”　　冯迪凡郭丽琴　　虚晃了两次之后，狼真的要...</td>
      <td>2017-08-14</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>特朗普欲对华贸易战？美专家：将是美经济倒退</td>
      <td>参考消息</td>
      <td>原标题：特朗普欲开展对华贸易战？美专家：这将是美国经济的倒退资料图：美国总统特朗普新华社...</td>
      <td>2017-08-15</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>美国对华301条款战一触即发 外交部：贸易战只会双输</td>
      <td>第一财经日报</td>
      <td>特朗普政府对华“301条款战”一触即发中美贸易战只会双输　　冯迪凡　　针对美国总统特朗普...</td>
      <td>2017-08-15</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 人为将文本分为6个阶段，标记为0-6
# 将每个阶段的文本拼接到一起，形成了六个period
period_1 = " ".join(list(sina_news.loc[sina_news.loc[:,"阶段"] == 0,"内容"]))
period_2 = " ".join(list(sina_news.loc[sina_news.loc[:,"阶段"] == 1,"内容"]))
period_3 = " ".join(list(sina_news.loc[sina_news.loc[:,"阶段"] == 2,"内容"]))
period_4 = " ".join(list(sina_news.loc[sina_news.loc[:,"阶段"] == 3,"内容"]))
period_5 = " ".join(list(sina_news.loc[sina_news.loc[:,"阶段"] == 4,"内容"]))
period_6 = " ".join(list(sina_news.loc[sina_news.loc[:,"阶段"] == 5,"内容"]))
```

### 2 文本分词处理


```python
def get_cut_result(text, stopWordsPath):
    """
    实现效果: 输入一段文本，返回分词后，重新组成的文本(需要给出停用词的路径)
    input:  
        text: 一段由文本组成的字符串 
        stopWordPath: 停用词文件路径
    output: 
        cutted_concated: 分词后，重新组成的长字符串
    """
    # 导入停用词表
    line = open(stopWordsPath, 'r', encoding="utf8").readline()
    stopwords = line.split(",")
    
    # 构造数字、字母pat
    pat = re.compile("[a-z0-9A-Z]+")

    result = []
    seg_list_1 = jieba.cut(period_1, cut_all=True) # 使用jieba进行分词    
    for seg in seg_list_1:        # 对分词结束后获得的list重新拼接
        pat_find = re.search(pat, seg)
        if seg not in stopwords and pat_find is None:  # 过滤掉停词和全部是pat的词汇
            seg = ''.join(seg.split()) #  首先对空格进行处理
            if (seg != '' and seg != "\n" and seg != "\n\n") :
                result.append(seg)
        cutted_concated = " ".join(result)
    return cutted_concated

# 对上述的6个period进行分词
concate_1 = get_cut_result(period_1, r"C:\Users\YHY\Desktop\stopWord.txt")
concate_2 = get_cut_result(period_2, r"C:\Users\YHY\Desktop\stopWord.txt")
concate_3 = get_cut_result(period_3, r"C:\Users\YHY\Desktop\stopWord.txt")
concate_4 = get_cut_result(period_4, r"C:\Users\YHY\Desktop\stopWord.txt")
concate_5 = get_cut_result(period_5, r"C:\Users\YHY\Desktop\stopWord.txt")
concate_6 = get_cut_result(period_6, r"C:\Users\YHY\Desktop\stopWord.txt")
```

    Building prefix dict from the default dictionary ...
    Loading model from cache C:\Users\YHY\AppData\Local\Temp\jieba.cache
    Loading model cost 1.006 seconds.
    Prefix dict has been built succesfully.
    


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-5-98b9082580e7> in <module>()
         32 concate_4 = get_cut_result(period_4, r"C:\Users\YHY\Desktop\stopWord.txt")
         33 concate_5 = get_cut_result(period_5, r"C:\Users\YHY\Desktop\stopWord.txt")
    ---> 34 concate_6 = get_cut_result(period_6, r"C:\Users\YHY\Desktop\stopWord.txt")
    

    <ipython-input-5-98b9082580e7> in get_cut_result(text, stopWordsPath)
         23             if (seg != '' and seg != "\n" and seg != "\n\n") :
         24                 result.append(seg)
    ---> 25         cutted_concated = " ".join(result)
         26     return cutted_concated
         27 
    

    KeyboardInterrupt: 


### 3 计算和输出tf idf值


```python
# 将分词的结果append到一个列表里，作为tf idf的输入
corpus = []
corpus.append(concate_1)
corpus.append(concate_2)
corpus.append(concate_3)
corpus.append(concate_4)
corpus.append(concate_5)
corpus.append(concate_6)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-6-7393d17583c7> in <module>()
          6 corpus.append(concate_4)
          7 corpus.append(concate_5)
    ----> 8 corpus.append(concate_6)
    

    NameError: name 'concate_6' is not defined



```python
# 初始化一个CountVectorizer类
# 对corpus里的文本计算tf idf值
vectorizer = CountVectorizer()    
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))

word = vectorizer.get_feature_names() #所有文本的关键字
weight = tfidf.toarray()              #对应的tfidf矩阵
```


```python
# 打印关键词的个数
print(len(word))     #关键词的个数
```


```python
# 观察第一阶段的tf idf
weight[0]
```


```python
# 将各个阶段的tf idf值、关键词等组合成一个字典
score_dict = {}
for i in range(len(corpus)):
    scores = weight[i]
    score_dict[str(i)] = {key:value for (key,value) in zip(scores,word)}
    # score_dict['0'] 这里的0表示的第几阶段
```


```python
# 输出各个阶段tf idf值排名前n的关键词
# 第一阶段的前10个关键词
top_30 = sorted(score_dict["0"].keys(),reverse=True)[0:30]
for i in range(30):
    print(score_dict["0"][top_30[i]] + ":" + str(top_30[i]))
```

--------------------------------------end--------------------------------------
