import pandas as pd #数据分析
import numpy as np #科学计算
from pandas import Series,DataFrame
from numpy import nan as NaN
import random


def silce_embarked(x):
    if x=='Q':
        return (1)
    elif x=='S':
        return(2)
    elif x=='C':
        return(3)
    else:
        return(random.randint(1,3))

def silce_cabin(x):
    if x!=0:
        return (1)
    else:
        return(0)

if __name__=='__main__':
    tr = pd.read_csv('dealtr_col1.csv')
    te = pd.read_csv('dealte_col1.csv')

    tr['Cabin'].fillna(value=0,inplace=True)
    tr['Cabin']=tr['Cabin'].map(lambda x:silce_cabin(x))
    te['Cabin'].fillna(value=0, inplace=True)
    te['Cabin'] = te['Cabin'].map(lambda x: silce_cabin(x))
    """
    tr['Embarked_Q'] = tr['Embarked'].map(lambda x: silce_embarked(x, 'Q'))
    tr['Embarked_S'] = tr['Embarked'].map(lambda x: silce_embarked(x, 'S'))
    tr['Embarked_C'] = tr['Embarked'].map(lambda x: silce_embarked(x, 'C'))

    te['Embarked_Q'] = te['Embarked'].map(lambda x: silce_embarked(x, 'Q'))
    te['Embarked_S'] = te['Embarked'].map(lambda x: silce_embarked(x, 'S'))
    te['Embarked_C'] = te['Embarked'].map(lambda x: silce_embarked(x, 'C'))
    """
    tr['Embarked1'] = tr['Embarked'].map(lambda x:silce_embarked(x))
    te['Embarked1'] = te['Embarked'].map(lambda x:silce_embarked(x))
    #tr.drop(['Embarked'], 1, inplace=True)
    #te.drop(['Embarked'], 1, inplace=True)

    tr.to_csv('silce_tr.csv',index=False)
    te.to_csv('silce_te.csv',index=False)
'''
    pd.set_option('display.max_row', 10)
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 1000)
    print(tr)
    print(te)
'''
