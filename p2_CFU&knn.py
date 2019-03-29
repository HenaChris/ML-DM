#-----------------基于KNN的协同过滤算法----------------------#
import pandas as pd
import time 
import math
from texttable import Texttable
import numpy as np

#定义导入文件函数
def readFile(file_name):        
    f = open(file_name,'r',encoding='utf-8')
    line = []
    line = f.readlines()
    f.close()
    return line

#生成电影信息表
"""
>>movies = readFile(filename2)
>>movies[:2]
["\ufeff1::Toy Story (1995)::Animation|Children's|Comedy\n", "2::Jumanji (1995)::Adventure|Children's|Fantasy\n"]
"""

#处理movies表格
def gen_movie_info(movies):
     movie_info = {}
     for movie in movies[1:]:
        arr = movie.split('::')
        movie_info[int(arr[0])]=arr[1:]#除去了第一行
     movie_info[1]=['Toy Story (1995)',"Animation|Children's|Comedy\n"]#特殊处理第一行，之前因为编码问题而转为TXT格式了。TXT格式的第一行不好分割。
     return movie_info

#生成电影信息字典
'''
>>movie_info = gen_movie_info(movies)
>> movie_info[12]
['Dracula: Dead and Loving It (1995)', 'Comedy|Horror\n']
'''

#生成用户评分表
'''
>>user = readFile(filename1)
>> user[:2]
['\ufeff1::1193::5::978300760\n', '1::661::3::978302109\n']
'''

#处理用户评分
def gen_temp_user(user):
    temp_user = []
    for line in user[1:]:
            rate = line.split('::')
            temp_user.append([int(rate[0]),int(rate[1]),int(rate[2])])
    temp_user.insert(0,[1,1193,5])#同上
    return temp_user

'''
>>temp_user = gen_temp_user(user)
>> temp_user[12]
[1, 2398, 4],iid为1的用户，对id为2398的电影评分为4
'''

#创建用户评分字典
def createUserRankDic(temp_user):
    user_rate_dict = {}
    item_to_user={}
    for i in temp_user:#对i[1]电影评为i[2]
        user_rank=(i[1],i[2])
        #用户和电影评分之间的字典
        if i[0] in user_rate_dict:                    #如果用户已经存在，则append，否则直接等
            user_rate_dict[i[0]].append(user_rank)
        else:
            user_rate_dict[i[0]] = [user_rank]
        #每一部电影和用户之相关的用户字典
        if i[1] in item_to_user:                #如果i[1]电影在用户的电影清单里面，则把用户的id加到以电影id为键的字典中
            item_to_user[i[1]].append(i[0])
        else:
            item_to_user[i[1]]=[i[0]]
    return user_rate_dict,item_to_user

#得到了用户-电影评分清单和电影-用户清单
'''
>> user_rate_dict,item_to_user = createUserRankDic(temp_user) #
>> user_rate_dict[12]#id为12的用户评分清单，对电影为id，1252的评分为3分
[(1252, 3), (3362, 3), (1193, 4), (1198, 5), (593, 5), (813, 3), (3897, 4), (2804, 5), (919, 5), (923, 5), (858, 5), (934, 2), (3658, 4), (1641, 3), (111, 5), (1221, 5), (3265, 4), (1303, 4), (1233, 3), (999, 4), (2616, 1), (3785, 3), (1247, 3)]
>>item_to_user[12]#对电影id为12的用户编号为90,177，195.......
[90, 117, 195, 202, 207, 237, 245, 302, 321, 424, 438, 531, 543, 549, 569, 660, 667, 678, 699, 714, 770, 777, 800, 808, 869, 881, 889, 967, 1010, 1016, 1017, 1019, 1096, 1100, 1109, 1112, 1120, 1146, ......]
'''

#计算userid与每个邻居的相关系数
def calcSimlaryCosDist(userid,neighbor):        #计算相关系数，userid是待判断的id，neighbor也是id
    user_rate = user_rate_dict[userid]
    neighbor_rate  = user_rate_dict[neighbor]

    x = []           #记录userid的评分
    y = []           #记录neighbor的评分

    for each_rate1 in user_rate:
       for each_rate2 in neighbor_rate:
          if each_rate1[0]==each_rate2[0]:
              x.append(each_rate1[1])
              y.append(each_rate2[1])

    if len(x)>5:#二者同时看过5部相同的电影，我们才计算其相关系数
        x =np.asarray(x)
        y = np.asarray(y)
        avg_x = np.mean(x)
        avg_y = np.mean(y)
        sum_xy = np.sum((x-avg_x)*(y-avg_y))
        sum_x2 = np.sum((x-avg_x)*(x-avg_x))
        sum_y2 = np.sum((y-avg_y)*(y-avg_y))
        sx_sy = math.sqrt(sum_x2*sum_y2)
        return abs(sum_xy / sx_sy)
    else:
        return 0.0

#找到邻居并计算其相关系数
def clacNearestNeighbor(userid,user_rate_dict,item_to_user,k):
    neighbors = []
    for item in user_rate_dict[userid]:   #找到用户字典中键为userid的值，item代表（电影id，评分）
        #在每一部电影之相关的用户中查找邻居
        for neighbor in item_to_user[item[0]]:   #item[0],电影id，找到电影字典中键为item[0]的值，即用户id
            if neighbor != userid and neighbor not in neighbors:
                neighbors.append(neighbor)                 #只要与用户看过同一部电影的都称为是用户的邻居，因为同一个邻居可能与该用户看过同几部电影，所以要加上neighbor not in neighbors这一条件，neighbor记录的也是id

    #计算相似度并输出
    neighbors_dist = []

    for neighbor in neighbors:                                          #计算与每个邻居的距离
        dist=calcSimlaryCosDist(userid,neighbor)   #输入[（电影id，评分）......]
        neighbors_dist.append([dist,neighbor])
    neighbors_dist.sort(reverse=True)
    return neighbors_dist[:k]#输出前K位邻居

#距离用户userid最近的30个邻居,相关系数最大的
'''
>>neighbors_dist = clacNearestNeighbor(1,user_rate_dict,item_to_user，30) #用户id为1，可以改变值
>> neighbors_dist[:3]
[[1.0, 3497], [0.9191450300180581, 410], [0.919145030018058, 1027], [0.9058216273156765, 2808], [0.8944271909999159, 5541]
'''

#计算每一部电影对用户的推荐程度大小
def recommend(neighbors_dist,userid):

    recommend_dict = {}

    for each_neighbor_dist in neighbors_dist:#每一位邻居的[dist,neighbor]
        neighbor_id = each_neighbor_dist[1]
        neighbor_movies = user_rate_dict[neighbor_id]#邻居的（电影id，评分）
        user_movies=[i[0] for i in user_rate_dict[userid]]#用户userid，用户的电影

        for movie in neighbor_movies:
            if movie[0] not in user_movies:#不在userid的观影列表内
                if movie[0] not in recommend_dict:#电影id
                       recommend_dict[movie[0]]=each_neighbor_dist[0]#{电影id,dist}
                else:
                      recommend_dict[movie[0]]+=each_neighbor_dist[0]#{电影id,dist+=}
    
    #建立推荐列表
    recommend_list = []

    for key in recommend_dict.keys():#电影id
        recommend_list.append([recommend_dict[key],key])#将字典转化为list，其中元素的第一项为推荐程度大小，第二项为电影的ID

    recommend_list.sort(reverse=True)#根据推荐的程度大小进行排序

    return recommend_list

'''
>>recommend_list = recommend(neighbors_dist，1)
[[14.49929317807284, 2858], [11.722279125524595, 858], [11.448594879095387, 2396], [11.173774374358965, 318], [11.055140365931594, 1196], [10.49531783273265, 1210], 
......
'''

#画表格
#recommend_list_id = [i[1] for i in recommend_list] #推荐的电影id
#neighbors_id = [i[1] for i in neighbors_dist]#邻居id
#item_to_user ，电影id，用户1，用户2……
def table_recommend(recommend_list_id,neighbors_id,item_to_user,movie_info):

    table = Texttable()
    table.set_deco(Texttable.HEADER)
    table.set_cols_dtype(['t',#text
                          't',#float (decimal)
                          't'])#automatic
    table.set_cols_align(["l",'l','l'])

    rows = []
    rows.append([u'movie name',u'type',u"form userid"])
    #输出前10个推荐项
    for movie_id in recommend_list_id[:10]:#找出推荐排名前十的电影
        from_user = []
        for user_id in item_to_user[movie_id]:#电影对应的用户
            if user_id in neighbors_id:#找谁推荐的电影
                from_user.append(user_id)
        rows.append([movie_info[movie_id][0],movie_info[movie_id][1],from_user[:5]])
    table.add_rows(rows)
    print(table.draw())

#输出表格
"""table_recommend(recommend_list_id,neighbors_id,item_to_user,movie_info)
"""


if __name__ == '__main__':
    userid = 1 #设定需要推荐的用户id
    k = 30 #计算邻居数
    filename1 = r'D:\论文研读\2.KNN算法\movielens_files\ml-1m\ratings.txt'
    filename2 = r'D:\论文研读\2.KNN算法\movielens_files\ml-1m\movies.txt'
    movies = readFile(filename2)
    movie_info = gen_movie_info(movies)
    user = readFile(filename1)
    temp_user = gen_temp_user(user)
    user_rate_dict,item_to_user = createUserRankDic(temp_user) #得到了用户-电影评分清单和电影-用户清单
    neighbors_dist = clacNearestNeighbor(userid,user_rate_dict,item_to_user,k) #用户id为1，可以改变值
    recommend_list = recommend(neighbors_dist,userid)#推荐列表
    recommend_list_id = [i[1] for i in recommend_list] #推荐的电影id
    neighbors_id = [i[1] for i in neighbors_dist]#邻居id
    table_recommend(recommend_list_id,neighbors_id,item_to_user,movie_info)

"""
table_recommend(recommend_list_id,neighbors_id,item_to_user,movie_info)
       movie name                     type                    form userid       
================================================================================
American Beauty (1999)      Comedy|Drama                [410, 789, 1027, 1133,  
                                                        2331]                   
Godfather, The (1972)       Action|Crime|Drama          [339, 876, 1027, 1133,  
                                                        1475]                   
Shakespeare in Love         Comedy|Romance              [410, 789, 1332, 1475,  
(1998)                                                  1574]                   
Shawshank Redemption, The   Drama                       [339, 1027, 1133, 1475, 
(1994)                                                  2266]                   
Star Wars: Episode V -      Action|Adventure|Drama|Sc   [339, 789, 876, 885,    
The Empire Strikes Back     i-Fi|War                    1133]                   
(1980)                                                                          
Star Wars: Episode VI -     Action|Adventure|Romance|   [339, 789, 876, 885,    
Return of the Jedi (1983)   Sci-Fi|War                  1133]                   
                                                                                
Godfather: Part II, The     Action|Crime|Drama          [339, 876, 1027, 1133,  
(1974)                                                  1475]                   
Being John Malkovich        Comedy                      [789, 2266, 2331, 2395, 
(1999)                                                  2808]                   
Silence of the Lambs, The   Drama|Thriller              [876, 1027, 1133, 2266, 
(1991)                                                  2391]                   
L.A. Confidential (1997)    Crime|Film-                 [876, 1133, 2266, 2331, 
                            Noir|Mystery|Thriller       2391]                       

"""


####---------------------------学习阶段-----------------------------------
##--------------------------数据预处理学习---------------------------------
#来源：https://blog.csdn.net/qq_40587575/article/details/81331717
#coding=utf-8
# MovieLens 1M数据集含有来自6000名用户对4000部电影的100万条评分数据。
# 分为三个表：评分，用户信息，电影信息。这些数据都是dat文件格式
# ，可以通过pandas.read_table将各个表分别读到一个pandas DataFrame对象中


start = time.clock()
filename1 =r'D:\论文研读\2.KNN算法\movielens_files\ml-1m\users.dat'
filename2 = r'D:\论文研读\2.KNN算法\movielens_files\ml-1m\ratings.dat'
filename3 = r'D:\论文研读\2.KNN算法\movielens_files\ml-1m\movies.dat'
pd.options.display.max_rows = 10
uname = ['user_id','gender','age','occupation','zip']
users = pd.read_table(filename1, sep='::', header = None, names=uname,encoding = 'cp936', engine='python')
print(users.head()) #年龄和职业都是使用编码的形式给出来的

print(users.shape)  # (6040, 5)
 
rnames = ['user_id','movie_id','rating','timestamp']
ratings = pd.read_table(filename2, encoding = 'cp936',header =None, sep='::',names=rnames, engine= 'python')
print(ratings.head())

mnames = ['movie_id','title','genres']  # genres 表示影片的体裁是什么
movies = pd.read_table(filename3, encoding = 'utf-8',header = None, sep='::', names = mnames, engine='python')
print(movies.head())
#年龄和职业编码
"""
- Age is chosen from the following ranges:
 
	*  1:  "Under 18"
	* 18:  "18-24"
	* 25:  "25-34"
	* 35:  "35-44"
	* 45:  "45-49"
	* 50:  "50-55"
	* 56:  "56+"
 
- Occupation is chosen from the following choices:
 
	*  0:  "other" or not specified
	*  1:  "academic/educator"
	*  2:  "artist"
	*  3:  "clerical/admin"
	*  4:  "college/grad student"
	*  5:  "customer service"
	*  6:  "doctor/health care"
	*  7:  "executive/managerial"
	*  8:  "farmer"
	*  9:  "homemaker"
	* 10:  "K-12 student"
	* 11:  "lawyer"
	* 12:  "programmer"
	* 13:  "retired"
	* 14:  "sales/marketing"
	* 15:  "scientist"
	* 16:  "self-employed"
	* 17:  "technician/engineer"
	* 18:  "tradesman/craftsman"
	* 19:  "unemployed"
	* 20:  "writer"
"""
#使用merge 函数将3个表进行合并
ratings_users = pd.merge(ratings, users)
data = pd.merge(ratings_users, movies)
print(data.head())
'''
   user_id  movie_id  ...                                   title  genres
0        1      1193  ...  One Flew Over the Cuckoo's Nest (1975)   Drama
1        2      1193  ...  One Flew Over the Cuckoo's Nest (1975)   Drama
2       12      1193  ...  One Flew Over the Cuckoo's Nest (1975)   Drama
3       15      1193  ...  One Flew Over the Cuckoo's Nest (1975)   Drama
4       17      1193  ...  One Flew Over the Cuckoo's Nest (1975)   Drama
'''

#分性别统计评分
#index  表示索引，values表示所要进行分析的数据， columns允许选择一个或多个列,以columns作为分组的列
mean_ratings = data.pivot_table(values ='rating', index='title', columns ='gender', aggfunc='mean')

#使用选择的数据进行统计分析
#过滤掉评分小于250人的电影
ratings_by_title = data.groupby('title').size()#分组统计
active_titles = ratings_by_title.index[ratings_by_title>=250]#找到索引
mean_ratings = mean_ratings.loc[active_titles]
print(mean_ratings[:5])
'''
gender                                    F         M
title                                                
'burbs, The (1989)                 2.793478  2.962085
10 Things I Hate About You (1999)  3.646552  3.311966
101 Dalmatians (1961)              3.791444  3.500000
101 Dalmatians (1996)              3.240000  2.911215
12 Angry Men (1957)                4.184397  4.328421
'''


#查看女性喜欢的电影
top_ratings_F = mean_ratings.sort_values(by = 'F',ascending = False)
print(top_ratings_F[:10])

'''
gender                                                     F         M
title                                                                 
Close Shave, A (1995)                               4.644444  4.473795
Wrong Trousers, The (1993)                          4.588235  4.478261
Sunset Blvd. (a.k.a. Sunset Boulevard) (1950)       4.572650  4.464589
Wallace & Gromit: The Best of Aardman Animation...  4.563107  4.385075
Schindler's List (1993)                             4.562602  4.491415
Shawshank Redemption, The (1994)                    4.539075  4.560625
Grand Day Out, A (1992)                             4.537879  4.293255
To Kill a Mockingbird (1962)                        4.536667  4.372611
Creature Comforts (1990)                            4.513889  4.272277
Usual Suspects, The (1995)                          4.513317  4.518248
'''

#查看男性观众喜欢的电影
top_ratings_M = mean_ratings.sort_values(by = 'M',ascending = False)
print(top_ratings_M[:10])

'''
gender                                                     F         M
title                                                                 
Godfather, The (1972)                               4.314700  4.583333
Seven Samurai (The Magnificent Seven) (Shichini...  4.481132  4.576628
Shawshank Redemption, The (1994)                    4.539075  4.560625
Raiders of the Lost Ark (1981)                      4.332168  4.520597
Usual Suspects, The (1995)                          4.513317  4.518248
Star Wars: Episode IV - A New Hope (1977)           4.302937  4.495307
Schindler's List (1993)                             4.562602  4.491415
Wrong Trousers, The (1993)                          4.588235  4.478261
Close Shave, A (1995)                               4.644444  4.473795
Rear Window (1954)            
'''


#计算男女差异最大的电影
mean_ratings['diff1'] = abs(mean_ratings['M'] - mean_ratings['F'])
sort_by_diff1 = mean_ratings.sort_values(by = 'diff1',ascending=False)
print(sort_by_diff1[:10])
'''
gender                                         F         M     diff1
title                                                               
Dirty Dancing (1987)                    3.790378  2.959596  0.830782
Good, The Bad and The Ugly, The (1966)  3.494949  4.221300  0.726351
Kentucky Fried Movie, The (1977)        2.878788  3.555147  0.676359
Jumpin' Jack Flash (1986)               3.254717  2.578358  0.676359
Dumb & Dumber (1994)                    2.697987  3.336595  0.638608
Longest Day, The (1962)                 3.411765  4.031447  0.619682
Cable Guy, The (1996)                   2.250000  2.863787  0.613787
Evil Dead II (Dead By Dawn) (1987)      3.297297  3.909283  0.611985
Grease (1978)                           3.975265  3.367041  0.608224
Hidden, The (1987)   
'''

#女性喜欢而男性不喜欢
mean_ratings['diff2'] = mean_ratings['M'] - mean_ratings['F']
sort_by_diff2 = mean_ratings.sort_values(by = 'diff2')
print(sort_by_diff2[:10])

#男性喜欢而女性不喜欢
sort_by_diff2[::-1][:10]

#计算得分数据的标准差，找出分歧最大的电影
rating_std = data.groupby('title')['rating'].std()
rating_std = rating_std.loc[active_titles]
print(rating_std.sort_values(ascending=False)[:10])
'''
print(rating_std.sort_values(ascending=False)[:10])
title
Dumb & Dumber (1994)                     1.321333
Blair Witch Project, The (1999)          1.316368
Natural Born Killers (1994)              1.307198
Tank Girl (1995)                         1.277695
Rocky Horror Picture Show, The (1975)    1.260177
Eyes Wide Shut (1999)                    1.259624
Evita (1996)                             1.253631
Billy Madison (1995)                     1.249970
Fear and Loathing in Las Vegas (1998)    1.246408
Bicentennial Man (1999)                  1.245533
Name: rating, dtype: float64
'''
end = time.clock()
spending_time = end -start
print("花费的时间为：%.2f"%spending_time+'s')

##------------------    -协同过滤学习--------------------------------------------------------------##
"""
协同过滤（Collaborative Filtering）字面上的解释就是在别人的帮助下来过滤筛选，协同过滤一般是在海量的用户中发现一小部分和你品味比较相近的，在协同过滤中，这些用户称为邻居，然后根据他们喜欢的东西组织成一个排序的目录来推荐给你。问题的重点就是怎样去寻找和你比较相似的用户，怎么将那些邻居的喜好组织成一个排序的目录给你，要实现一个协同过滤的系统，需要以下几个步骤：

1.计算其他用户和你的相似度，可以使用反差表忽略一部分用户
2.根据相似度的高低找出K个与你最相似的邻居
3.在这些邻居喜欢的物品中，根据邻居与你的远近程度算出每一件物品的推荐度
4.根据每一件物品的推荐度高低给你推荐物品。
--------------------- 
作者：秋水长天q 
来源：CSDN 
原文：https://blog.csdn.net/Augster/article/details/53352653 
"""
import math
from texttable import Texttable

def calcSimlaryCosDist(user1,user2):        #计算相关系数，user1是待判断的，user2是neighbor
    sum_x = 0.0
    sum_y = 0.0
    sum_xy = 0.0
    avg_x = 0.0
    avy_y = 0.0
    for key in user1:
        avg_x += key[1] #key[1]评分
    avg_x = avg_x/len(user1)

    for key in user2:
        avg_y += key[1]
        avg_y = avg_y / len(user2)

    for key1 in user1:
        for key2 in user2:
            if key1[0] == key2[0]:  #看了同一部电影才算
                sum_xy += (key1[1] - avg_x)*(key2[1] - avg_y)
                sum_y += (key2[1] - avg_y)*(key2[1] - avg_y)
        sum_x += (key1[1] - avg_x)*(key1[1] - avg_x)
   
    if sum_xy == 0.0:
        return 0.0
    sx_sy = math.sqrt(sum_x*sum_y)
    return sum_xy / sx_sy

def readFile(file_name):        #导入文件
    f = open(file_name,'r',encoding='gb18030')
    line = []
    line = f.readlines()
    f.close()
    return line

#读取电影信息，返回电影的字典，key值为电影id,value值为电影信息
def getMoviesList(file_name):
    lines = readFile(file_name)
    movie_info = {}
    for movie in lines:
        arr=movie.spilt('|')
        movie_info[int(arr[0])]=arr[1:]
    return movie_info

#将rating文件中的信息转化为数组形式
#返回用户ID，电影ID，评分，时间戳
def getRatingInformation(ratings):
    r = []
    for line in ratings:
        rate = line.split('\t')
        r.append([int(rate[0]),int(rate[1]),int(rate[2])])
    return r

#生成用户评分的数据结构
#输入：[[2,1,5],[2,4,2]...]，用户2对电影1的评分是5分
#输出：用户打分字典和电影与值打分关联用户的字典
#rate_dic[2]=[(1,5),(4,2)].... 表示用户2对电影1的评分是5，对电影4的评分是2
#item_to_user[661]=[1,23,49....]
def createUserRankDic(rates):
    user_rate_dict = {}
    item_to_user={}
    for i in rates:#对i[1]电影评为i[2]
        user_rank=(i[1],i[2])
        #用户和电影评分之间的字典
        if i[0] in user_rate_dict:                    #如果用户已经存在，则append，否则直接等
            user_rate_dict[i[0]].append(user_rank)
        else:
            user_rate_dict[i[0]] = [user_rank]
        #每一部电影和用户之相关的用户字典
        if i[1] in item_to_user:                #如果i[1]电影在用户的电影清单里面，则把用户的id加到以电影id为键的字典中
            item_to_user[i[1]].append(i[0])
        else:
            item_to_user[i[1]]=[i[0]]
    return user_rate_dict,item_to_user

#计算与制定的邻居之间最为相近的邻居
#输入：指定的用户ID，用户对电影的评分表，电影对应的用户表
#输出：与制定用户最为相邻的邻居列表
#    1.用户字典：dic[用户id]=[(电影id,电影评分)...]
#    2.电影字典：dic[电影id]=[用户id1,用户id2...]
def clacNearestNeighbor(userid,user_dict,item_dict):
    neighbors = []
    for item in user_dict[userid]:                       #找到用户字典中键为userid的值，item代表（电影id，评分）
        #在每一部电影之相关的用户中查找邻居
        for neighbor in item_dict[item[0]]:               #item[0],电影id，找到电影字典中键为item[0]的值，即用户id
            if neighbor != userid and neighbor not in neighbors:
                neighbors.append(neighbor)                 #只要与用户看过同一部电影的都称为是用户的邻居，因为同一个邻居可能与该用户看过同几部电影，所以要加上neighbor not in neighbors这一条件，neighbor记录的也是id

    #计算相似度并输出
    neighbors_dist = []
    for neighbor in neighbors:                                          #计算与每个邻居的距离
        dist=calcSimlaryCosDist(user_dict[userid],user_dict[neighbor])   #输入[（电影id，评分）......]
        neighbors_dist.append([dist,neighbor])
    neighbors_dist.sort(reverse=True)
    return neighbors_dist

def  recommendationByUserFC(file_name,userid,k=5):
    test_contents=readFile(file_name)#读取文件
    test_rates=getRatingInformation(test_contents)#得到用户电影评分之间标准数据
   #格式化成字典数据
   #1.用户字典：dict[用户ID]=[（电影ID，电影评分）……]
   #2.电影字典:dict[电影ID]=[用户ID1，用户ID2]
    test_dict,test_item_to_user = createUserRankDic(test_rates)
   #计算与userid最为相近的前k个用户，返回数组格式为[[相似度，用户id]
    neighbors=calcNearestNeighbor(userid,test_dict,test_item_to_user)[:k] #sort过了
   #计算邻居的每一部电影与被推荐用户之间的相似度大小
    recommend_dict={}
    for neighbor in neighbors:#[dist,neighbor]
       neighbor_user_id = neighbor[1]
       movies = test_dict[neighbor_user_id]
       #计算每一部电影对用户的推荐程度大小
       for movie in movies:
           if movie[0] not in recommend_dict:
               recommend_dict[movie[0]]=neighbor[0]
           else:
                recommend_dict[movie[0]]+=neighbor[0]

#建立推荐列表
    recommend_list=[]
    for key in recommend_dict:
        recommend_list.append([recommend_dict[key],key])#将字典转化为list，其中元素的第一项为推荐程度大小，第二项为电影的ID
    recommend_list.sort(reverse=True)#根据推荐的程度大小进行排序
    user_movies=[i[0] for i in test_dict[userid]]#userid用户评分过的所有电影
    return [i[1] for i in recommend_list], user_movies, test_item_to_user, neighbors

if __name__=="__main__":
    movies = getMoviesList('u.item')#获取电影列表
    recommend_list,user_movie,items_movie,neighbors=recommendationByUserFC('u.data',1,80)
    neighbors_id = [i[1] for i in neighbors]
    table = Texttable()
    table.set_deco(Texttable.HEADER)
    table.set_cols_dtype(['t',#text
                          't',#float (decimal)
                          't'])#automatic
    table.set_cols_align(["l",'l','l'])
    rows = []
    rows.append([u'movie name',u'release',u"form userid"])
    #输出前20个推荐项
    for movie_id in recommend_list[:20]:
        from_user = []
        for user_id in items_movie[movie_id]:
            if user_id in items_movie[movie_id]:
                if user_id in neighbors_id:
                    from_user.append(user_id)
        rows.append([movies[movie_id][0],movies[movie_id][1],from_user[:3]])
        table.add_rows(rows)
        print(table.draw())

