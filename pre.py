import pandas as pd #数据分析
import numpy as np #科学计算
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

if __name__ == '__main__':
    data_train = pd.read_csv("train1.csv")
    #print(data_train.columns)#显示属性
    #data_train.info()#显示简单统计
"""
    pd.set_option('display.max_row',10)
    pd.set_option('display.max_columns',20)
    pd.set_option('display.width',1000)
    print(data_train.describe())
    # 显示数值属性统计结果
"""

#"""
fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数
#plt.subplot2grid((2,3),(0,0))             # 在一张大图里分列几个小图
data_train.Survived.value_counts().plot(kind='bar')# 所有获救情况统计表
plt.title(u"Rescue (1 is survived)")
plt.ylabel(u"NumofHuman")
plt.show()


fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数
data_train.Pclass.value_counts().plot(kind="bar")
plt.ylabel(u"NumofHuman")
plt.title(u"Pclass")#各等级的票与生还的统计
plt.show()


fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数
plt.scatter(data_train.Survived, data_train.Age)
plt.ylabel(u"Age")                         # y轴为年龄
plt.grid(b=True, which='major', axis='y') 
plt.title(u"RescueByAge (1 Survived)")#是否生还和年龄的统计图
plt.show()


fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数
data_train.Age[data_train.Pclass == 1].plot(kind='kde')   
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel(u"Age")# plots an axis lable
plt.ylabel(u"Density")#y轴为各票等级在年龄上的密度分布
plt.title(u"AgeByPclass")
plt.legend((u'Pclass1', u'Pclass2',u'Pclass3'),loc='best')
plt.show()


fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数
data_train.Embarked.value_counts().plot(kind='bar')
plt.title(u"NumByEmbarked")
plt.ylabel(u"NumofHuman")
plt.show()
#"""

#"""
fig = plt.figure()
fig.set(alpha=0.2)#颜色参数
Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df=pd.DataFrame({u'Survived':Survived_1, u'Unsurvived':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"RescueByPclass")
plt.xlabel(u"Pclass")
plt.ylabel(u"NumofHuman")#通过票的等级对是否生还作统计
plt.show()
"""

"""
fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数
Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
df=pd.DataFrame({u'Survived':Survived_1, u'Unsurvived':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"RescueByEmbarked")
plt.xlabel(u"Embarked")
plt.ylabel(u"NumofHuman")#登船港口和生还的统计
plt.show()
"""
"""
fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数
Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
df=pd.DataFrame({u'Male':Survived_m, u'Female':Survived_f})
df.plot(kind='bar', stacked=True)
plt.title(u"RescueBySex(1 Survived)")
plt.xlabel(u"Sex")
plt.ylabel(u"NumofHuman")#根据性别统计生还情况
plt.show()
"""
"""
fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数
Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
df=pd.DataFrame({u'Have':Survived_cabin, u'NotHave':Survived_nocabin}).transpose()
df.plot(kind='bar', stacked=True)
plt.title(u"RescueByWhetherCabin")#是否有房间号与生还情况的统计
plt.xlabel(u"whetherCabin")#是否有房间号
plt.ylabel(u"NumofHuman")
plt.show()
"""

"""
fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数
Sur_Mr = data_train.Survived[data_train.Title == 'Mr'].value_counts()
Sur_Mrs = data_train.Survived[data_train.Title == 'Mrs'].value_counts()
Sur_Miss = data_train.Survived[data_train.Title == 'Miss'].value_counts()
Sur_Master = data_train.Survived[data_train.Title == 'Master'].value_counts()
df=pd.DataFrame({u'Mr':Sur_Mr, u'Mrs':Sur_Mrs, u'Miss':Sur_Miss, u'Master':Sur_Master})
df.plot(kind='bar', stacked=True)
plt.title(u"RescueByTitle(1 Survived)")
plt.xlabel(u"Title")
plt.ylabel(u"NumofHuman")#根据称谓统计生还情况
plt.show()
"""



"""
fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数
Survived_0 = data_train.Title[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Title[data_train.Survived == 1].value_counts()
df=pd.DataFrame({u'Survived':Survived_1, u'Unsurvived':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"RescueByTitle")
plt.xlabel(u"Title")
plt.ylabel(u"NumofHuman")#登船港口和生还的统计
plt.show()
"""
"""
fig = plt.figure()
fig.set(alpha=0.2)
data_train.Age[data_train.Title == 'Mr'].plot(kind='kde')
data_train.Age[data_train.Title == 'Mrs'].plot(kind='kde')
data_train.Age[data_train.Title == 'Miss'].plot(kind='kde')
data_train.Age[data_train.Title == 'Master'].plot(kind='kde')
plt.xlabel(u"Age")# plots an axis lable
plt.ylabel(u"Density")#y轴为各称谓在年龄上的密度分布
plt.title(u"AgeByTitle")
plt.legend((u'Mrs', u'Mr',u'Miss',u'Master'),loc='best')
plt.show()
#"""


