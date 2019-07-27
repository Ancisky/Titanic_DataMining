import pandas as pd #数据分析
import numpy as np #科学计算
from pandas import Series,DataFrame
from sklearn.ensemble import RandomForestRegressor

'''
def set_missing_ages2(data):
    # 把已有的数值型特征放入Random Forest Regressor中
    tr = data[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = tr[tr.Age.notnull()].as_matrix()
    unknown_age = tr[tr.Age.isnull()].as_matrix()

    pd.set_option('display.max_row', 10)
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 1000)
    # print(data.loc[(data.Age.isnull())])
    # print('\n\nend\n')

    # y即目标年龄
    y1 = known_age[:, 0]

    # X即特征属性值
    X1 = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr2 = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr2.fit(X1, y1)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr2.predict(unknown_age[:, 1::])

    # 用得到的预测结果填补原缺失数据
    data.loc[(data.Age.isnull()), 'Age'] = predictedAges

    return data
    '''
from sklearn.ensemble import RandomForestRegressor

def set_missing_ages1(data):
    # 把已有的数值型特征放入Random Forest Regressor中
    tr = data[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass','Title1']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = tr[tr.Age.notnull()].as_matrix()
    unknown_age = tr[tr.Age.isnull()].as_matrix()

    pd.set_option('display.max_row', 10)
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 1000)
    # print(data.loc[(data.Age.isnull())])
    # print('\n\nend\n')

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr1 = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr1.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr1.predict(unknown_age[:, 1::])

    # 用得到的预测结果填补原缺失数据
    data.loc[(data.Age.isnull()), 'Age'] = predictedAges

    return data,rfr1

if __name__ == '__main__':
    data_tr=pd.read_csv('midtr.csv')
    data_te = pd.read_csv('midte.csv')
    train=[data_tr,regressor]=set_missing_ages1(data_tr)
    #train存放拟合过的数据和训练出来的模型

    tmp = data_te[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass','Title1']]
    null_age = tmp[data_te.Age.isnull()].as_matrix()
    # 根据特征属性X预测年龄并补上
    X = null_age[:, 1:]
    predictedAges = train[1].predict(X)
    data_te.loc[(data_te.Age.isnull()), 'Age'] = predictedAges

    #print(train[1])
    #data_tr.drop(['Mr','Mrs','Miss','Master'],axis=1,inplace=True)
    #data_te.drop(['Mr','Mrs','Miss','Master'], axis=1, inplace=True)
    #data_tr.to_csv('alldat_tr.csv',index=False)
    #data_te.to_csv('alldat_te.csv',index=False)
