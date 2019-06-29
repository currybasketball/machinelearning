# K-Nearest Neighbor
K近邻算法是选择距离最近的K个邻居的算法。根据K个邻居来分类，或者判断属于某种类别的概率。
**优点**：算法简单，便于理解。
**缺点**：需要比较所有数据，计算量较大，可能出现维度爆炸，KNN其实没有训练模型，每次只是计算一堆数据

## 使用sklearn练习KNN
```
from sklearn import datasets #引入sklearn数据
from sklearn.model_selection import train_test_split #用于拆分数据，可以指定随机种子，也可以使用默认的random值
from sklearn.neighbors import KNeighborsClassifier #引入KNN分类器
import numpy as np #引入numpy包，方便数据处理数据

iris = datasets.load_iris() #引入iris数据，这个是根据花的4个维度数据，去判断花的类型
X=iris.data # X 是花的特征向量
y=iris.target # y是花的种类
#拆分 random_state=2003 可以不指定吧，这样执行多次结果都不一样
X_train,X_test,y_train,y_test = train_test_split(X,y)
#建立模型 weights='uniform'(按照距离可以计算)/可以选择weights='distance'(按照半径计算)
KNC = KNeighborsClassifier(n_neighbors=3,weights='distance')# n_neightbors 唯一超参数，指定获取最近的K个点
KNC.fit(X_train,y_train) # 训练 计算距离
result = KNC.predict(X_test) # 使用测试数据计算
#计算精确度
from sklearn.metrics import accuracy_score #计算准确率
test = accuracy_score(result,y_test)
print(test)
print(np.count_nonzero(result==y_test)/len(X_test))
>>0.9736842105263158
>>0.9736842105263158
```
## 自己实现简单KNN
```
# 计算两个点之间距离 array[0]-array[1]
# (x1,y1)-(x2,y2): instance1 - instance2=(x1-x2,y1-y2)
# (instance1 - instance2)**2 = ((x1-x2)^2,(y1-y2)^2)
# sum((instance1 - instance2)**2) = (x1-x2)^2+(y1-y2)^2
def euc_dis(instance1,instance2):
    return np.sqrt(sum((instance1 - instance2)**2))

def knn_classify(X,y,testInstance,k):
    '''
    X:训练数据特征
    y:训练数据标签
    testInstance: 测试数据
    k: 选择多少个neighbors
    '''
    # 计算测试数据和训练数据之间每个点的距离，循环训练数据
    distances = [euc_dis(x,testInstance) for x in X]
    # 排序返回前K 个点
    kneighbors = np.argsort(distances)[:k]# 返回前K个符合的值
    count = Counter(y[kneighbors])
    # count是一个2*2矩阵
    return count.most_common()[0][0]
# 循环测试数据，调用方法
predict=[knn_classify(X_train,y_train,data,3) for data in X_test]
#计算准确率
result1=accuracy_score(predict,y_test)
print(result1)
print(np.count_nonzero((predict==y_test)==True)/len(X_test))
>0.9473684210526315
>0.9473684210526315
```
## one_hot 编码处理数据
在数据分析中，经常会碰到非数字的情况。比如星期，英文字母等。**sklearn** 提供了预处理方法，OneHotEncoder,LabelEncoder 
···
逻辑如下：
1. 使用LabelEncoder 把非数字转换为数字
2. 使用OneHotEncoder 把数字转换为多维数据
···
```
import pandas as pd
from numpy import argmax
import matplotlib 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# 引入预处理方法
from sklearn.preprocessing import OneHotEncoder,LabelEncoder 
df = pd.read_csv("data.csv")
label_encoder = LabelEncoder()
df["Color"]
```
0      blue
1      blue
2     black
3       red
4      grey
5       red
6     green
7     green
8     white
9     green
10    black
11    black
12    black
Name: Color, dtype: object
```
lnteger_encoded = label_encoder.fit_transform(df["Color"])
```
[[1]
 [1]
 [0]
 [4]
 [3]
 [4]
 [2]
 [2]
 [5]
 [2]
 [0]
 [0]
 [0]]
```
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = lnteger_encoded.reshape(len(lnteger_encoded),1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
onehot_encoded
```
[[0. 1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 1. 0.]
 [0. 0. 0. 1. 0. 0.]
 [0. 0. 0. 0. 1. 0.]
 [0. 0. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0.]
 [0. 0. 0. 0. 0. 1.]
 [0. 0. 1. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0.]]
