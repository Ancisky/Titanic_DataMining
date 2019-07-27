import pandas as pd #数据分析
import numpy as np #科学计算
from pandas import Series,DataFrame

if __name__ == '__main__':
    tr = pd.read_csv('silce_tr.csv')
    te = pd.read_csv('silce_te.csv')

    tr['Family'] = tr['SibSp'].map(lambda x:x)
    tr['Family'] += tr['Parch'].map(lambda x:x)
    te['Family'] = te['SibSp'].map(lambda x: x)
    te['Family'] += te['Parch'].map(lambda x: x)

    tr['Fare'].fillna(0,inplace=True)
    te['Fare'].fillna(0,inplace=True)

    tr.drop(['SibSp','Parch','Ticket'],axis=1,inplace=True)
    te.drop(['SibSp', 'Parch', 'Ticket'], axis=1, inplace=True)

    tr.to_csv('dealtr_final.csv',index=False)
    te.to_csv('dealte_final.csv',index=False)