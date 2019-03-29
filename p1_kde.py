核密度估计
#coding=utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import quad
import scipy.stats


#导入数据
path1 = "D:/论文研读/指数文件183545074/Index.csv"
index = pd.read_csv(path1,sep=',')

#计算收益率

r = []

for  i in range(len(index.Clsindex)-1):
    r.append(math.log(index.Clsindex[i+1]) - math.log(index.Clsindex[i]))

rtData = pd.DataFrame(r,columns = ['rt'])


#收益率描述性统计
describe_rt =  rtData.rt.describe()
describe_rt
#收益率散点图
rtData.plot()
plt.show()

#收益率直方图
rtData.hist()
plt.show()

#指数散点图
index.Clsindex.plot()
plt.show()

#指数直方图
index.Clsindex.hist()
plt.show()

#收益率的的核密度估计

def ourkde(u0,rt,h):

    fu = []

    for u in u0:

        t=(rt-u)/h

        K = h**(-1)*np.exp(-0.5*(t**2))/np.sqrt(2*math.pi)
          
        fu.append(np.mean(K))

    np.asarray(fu)

    return fu


#比较不同的h值得到的函数图像
n = len(rtData.rt)

u = np.linspace(-0.053647,0.054495,n)

h1 = 0.001
h2 = 0.002
h3 = 0.003
h4 = 0.004
h5 = 0.005
h6 = 0.008

#最优窗口
h = 1.06*np.std(rtData.rt)*n**(-1/5)

hlist = [h1,h2,h3,h4,h5,h6,h]

#h越大越平滑
for h0 in hlist:

    fu = ourkde(u,rtData.rt,h0)

    plt.plot(u,fu,'-')

    plt.show()



"""
#求核密度估计的另一种方法
def predkde(x0,rt,h):
    
    t = (rt - x0)/h

    k = h**(-1)*np.exp(-0.5*(t**2))/np.sqrt(2*math.pi)

    fx = np.mean(k)

    return fx


fxlist = []

for x0 in u:

    fxlist.append(predkde(x0,rtData.rt,h))

fxlist

fu = ourkde(u,rtData.rt,h)

fu1=np.asarray(fxlist)

sorted(fu)
sorted(fu1)
plt.plot(u,fu1,'g-')
plt.plot(u,fu,'r-')
plt.show()
"""


#求概率密度函数的均值,方差

lam1 = lambda x:x*np.mean(h**(-1)*np.exp(-0.5*(((rtData.rt - x)/h)**2))/np.sqrt(2*math.pi))

lam2 = lambda x:x**2*np.mean(h**(-1)*np.exp(-0.5*(((rtData.rt - x)/h)**2))/np.sqrt(2*math.pi))

Ex,err1 = quad(lam1,-np.inf,np.inf)

Ex2,err1 = quad(lam2,-np.inf,np.inf)

var = Ex2 - Ex**2

print("核密度估计均值为",Ex)

print("核密度估计方差为",var)

print("真实的均值为",np.mean(rtData.rt))

print("真实的方差为",np.var(rtData.rt))

#拟合优度的卡方统计量检验
#分箱统计对数收益率
#分箱边界
bins = [-np.inf,-0.025,-0.01,-0.005,0,0.005,0.01,0.025,np.inf]
cats = pd.cut(rtData['rt'],bins)

#categorical对象
cats

#统计频数
cut = pd.value_counts(cats,ascending=True)
cut

#计算频率
freq = []
for each in cut:
    freq.append(round((each/sum(cut)),3))

freq = np.asarray(freq)

#求分箱区间的概率值
m = 8
k = 1
n = len(rtData.rt)
bins = [-np.inf,-0.025,-0.01,-0.005,0,0.005,0.01,0.025,np.inf]
cut
f0 = [9,37,33,51,39,22,43,7]

#计算卡方统计量的值
def chi_test(m,k,n,bins,f0):

    chidict = {}

    for h in np.arange(0.003,0.008,0.0005):

        quadList = []

        lam = lambda x:np.mean(h**(-1)*np.exp(-0.5*(((rtData.rt - x)/h)**2))/np.sqrt(2*math.pi))

        for i in range(len(bins)-1):
            quadList.append(quad(lam,bins[i],bins[i+1])[0])
        
        quadArray = np.asarray(quadList)

        chi = 0
        for i in range(len(quadArray)):
            chi += (f0[i] - n*quadArray[i])**2/(n*quadArray[i])

        chidict[h]=chi

    return chidict

chidict = chi_test(m,k,n,bins,f0)

#计算卡方分布0.95分位数
α= 0.05
chi0 = scipy.stats.chi2.ppf(1-α,m-k-1)

#找临界的h
for v in chidict.values():
    if v > chi0:
        for key,val in chidict.items():
            if val == v:
                print('%.3f'%key)

#比较临界的h与最佳的h之间的函数图像差异
fu1 = ourkde(u,rtData.rt,h)
fu2 = ourkde(u,rtData.rt,h6)
plt.plot(u,fu1,'r-')
plt.plot(u,fu2,'b-')
plt.show()
