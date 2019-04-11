
#### 1.数据预处理

~~~python
##encoding='utf-8'
## stock return data

import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import os

###数据预处理，包括弥补缺失值（有些公司的股票信息不完整，有些日期没有）、批量导入数据
########## obtain full index of dates

index_path = r'D:\研一下\data mining\data\SZ399300.TXT'
index300 = pd.read_table(index_path,\
    encoding = 'cp936',header = None)
index2 = index300[:-1] #去除'1340  数据来源:通达信'行
index2.columns = ['date','o','h','l','c','v','to']
index2.index = index2['date'] #日期
~~~

~~~python
index2.index output:
    
'''
              date        o        h  ...        c            v            to
date                                  ...                                    
20100602  20100602  2729.33  2757.91  ...  2757.53   39587020.0  4.636423e+10
20100603  20100603  2769.10  2787.51  ...  2736.08   39587258.0  4.714326e+10
20100604  20100604  2721.36  2748.48  ...  2744.39   32746291.0  3.985836e+10
20100607  20100607  2692.83  2716.40  ...  2695.72   39541993.0  4.706270e+10
20100608  20100608  2694.11  2720.59  ...  2699.34   35140182.0  4.267710e+10
20100609  20100609  2711.64  2787.82  ...  2782.13   60209533.0  7.071008e+10
20100610  20100610  2755.94  2776.26  ...  2750.02   42906334.0  5.031225e+10
20100611  20100611  2766.20  2782.09  ...  2758.87   40558585.0  4.887148e+10
........................
'''
~~~

~~~python
#批量读入TXT文件
stock_path = r'D:\研一下\data mining\data\hs300-2\hs300'
names = os.listdir(stock_path)
close = []

for name in names:
    spath = stock_path + '\\' + name
    tmp = pd.read_table(spath,encoding = 'cp936',header = None)
    df = tmp[:-1]
    df.columns = ['date','o','h','l','c','v','to']
    df.index = df['date']
    df1 = df.reindex(index2.index, method = 'ffill')
    df2 = df1.fillna(method = 'bfill')
    close.append(df2['c'].values) #提出收盘价

close = np.asarray(close).T #(1340,600)1340天，600股票
retx = (close[1:,:] - close[0:-1,:])/close[:-1,:]
~~~

~~~python
retx output:
'''
array([[-0.01426873, -0.00938967, -0.00371058, ...,  0.        ,
         0.        ,  0.        ],
       [-0.00241255, -0.007109  ,  0.0018622 , ...,  0.        ,
         0.        ,  0.        ],
       [-0.03748489, -0.02386635, -0.00929368, ...,  0.        ,
         0.        ,  0.        ],
       ...,
       [ 0.03955104, -0.00569801,  0.00666032, ...,  0.07783641,
         0.01405923,  0.07756678],
       [ 0.03341902, -0.02005731, -0.01701323, ..., -0.03029376,
        -0.03038348, -0.04241338],
       [-0.04328358,  0.        , -0.02307692, ..., -0.02366677,
        -0.00730149,  0.01684342]])
'''
~~~

#### 2.主成分分析

~~~python
## PCA主成分分析
covx = np.cov(retx.T)
import numpy.linalg as la
u,v = la.eig(covx) # u-eigen value，v eigen vectors

~~~

##### 2.1均值mu

~~~python
#均值mu
mu = np.mean(retx,axis = 0)
plt.plot(mu)
plt.show()                     
~~~

![mu_Figure_1](https://github.com/HenaChris/-/blob/master/mu_Figure_1.png?raw=true)

<center>    图1  股票收益率均值



上图为300只股票收益率的均值，代表的含义为每只股票的平均收益率

##### 2.2特征向量

~~~python
#特征向量v1
plt.plot(v[:,0:1])
plt.show()
~~~

![v1_Figure_2](https://github.com/HenaChris/-/blob/master/v1_Figure_2.png?raw=true)

<center>图2  第一个特征向量

~~~python
#v1前五个最大值对应的股票
id1 = v[:,0].argsort()[-5:]
[names[a] for a in id1]
~~~

~~~python
output:
'''
['SZ000937.txt', 'SZ000686.txt', 'SZ002008.txt', 'SH601699.txt', 'SZ000559.txt']
'''
~~~



<center>   表1   v1前五个最大值对应的股票

| 股票代码 |   股票名称   |
| :------: | :----------: |
| SZ000937 | **冀中能源** |
| SZ000686 | **东北证券** |
| SZ002008 | **大族激光** |
| SH601699 | **潞安能源** |
| SZ000559 | **万向钱潮** |



~~~python
#特征向量v2
plt.plot(v[:,0:1])
plt.show()
~~~

![v2_Figure_3](https://github.com/HenaChris/-/blob/master/v2_Figure_3.png?raw=true)

<center>  图3  第二个特征向量

~~~python
#v2前五个最大值对应的股票
id2 = v[:,1].argsort()[-10:]
[names[a] for a in id2]
~~~

~~~python
output:
'''
['SZ002142.txt', 'SH601818.txt', 'SH601688.txt', 'SH600837.txt', 'SH600030.txt', 'SH600000.txt', 'SH600999.txt', 'SH600015.txt', 'SH600048.txt', 'SH601166.txt']
'''
~~~

 <center>  表2   v2前五个最大值对应的股票
 </center>

| 股票代码 | 股票名称 |
| :------: | :------: |
| SZ002142 | 宁波银行 |
| SH601818 | 光大银行 |
| SH601688 | 华泰证券 |
| SH600837 | 海通证券 |
| SH600030 | 中信证券 |
| SH600000 | 浦发银行 |
| SH600999 | 招商证券 |
| SH600015 | 华夏银行 |
| SH600048 | 保利地产 |
| SH601166 | 兴业银行 |

由表2可以看出特征向量v1集中在能源领域，v2较大值都集中在银行券商行业。唯一的例外是保利地产，待研究。

##### 2.3主成分得分

第一主成分得分衡量的是每个交易日市场的收益率综合表现。比较图5与图6，二者的波动形状较为一致，但总体上，第一主成分”放大了“这种波动。第一主成分得分较高的有['20150716', '20150529', '20120116', '20150826', '20150827', '20150710', '20150629', '20150709', '20150708', '20150915']等日期。大概集中在2015年5月到9月。从沪深300月k线图来看2015年5月至9月是股市下跌幅度较为剧烈。比较图4和图5,相应地，股市震荡会影响各个股票的收益率有较大的振幅。



![1554888951552](https://github.com/HenaChris/-/blob/master/hs300.png?raw=true)

<center> 图4 hs300月k线图

~~~python
#每日收益率均值
mu2 = np.mean(retx,axis=1)
plt.plot(mu2)
plt.show()
~~~

![mu_2Figure_1](https://github.com/HenaChris/-/blob/master/mu_2Figure_1.png?raw=true)

<center>图5   每日收益率均值

~~~python
#第一主成分得分
f1 = (retx - mu).dot(v[:,0:1])
~~~

![f1_Figure_1](https://github.com/HenaChris/-/blob/master/f1_Figure_1.png?raw=true)

<center> 图6  第一主成分得分

~~~python
id3 = f1.ravel().argsort()[-10:]
[index2.index[a] for a in id3]
~~~

~~~python
output:
'''
['20150716', '20150529', '20120116', '20150826', '20150827', '20150710', '20150629', '20150709', '20150708', '20150915']
'''
~~~

第二主成分得分图像与每日均值收益率的图像并不一致。说明第二主成分并不反应市场综合收益率。观察第二主成分得分较高的日期——['20150831', '20101014', '20141219', '20150626', '20141215', '20150703', '20150706', '20101008', '20150605', '20131129']

~~~python
#第二主成分得分
f2 = (retx - mu).dot(v[:,1:2])
~~~

![f2_Figure_1](https://github.com/HenaChris/-/blob/master/f2_Figure_1.png?raw=true)

<center> 图7 第二主成分得分

~~~python
id4 = f2.argsort()[-10:]
[index2.index[a] for a in id4]
~~~

~~~python
output:
'''
['20150831', '20101014', '20141219', '20150626', '20141215', '20150703', '20150706', '20101008', '20150605', '20131129']
'''
~~~

