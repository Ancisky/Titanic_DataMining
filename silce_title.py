import pandas as pd #数据分析
import numpy as np #科学计算
from pandas import Series,DataFrame

def silce_title(x,title):
    if x == title:
        return (1)
    else:
        return (0)

def silce_title1(x):
    if x=='Mr':
        return (4)
    elif x =='Mrs':
        return (3)
    elif x=='Miss':
        return (2)
    elif x=='Master':
        return (1)
    else:
        return (x)

if __name__ == '__main__':
    data_tr = pd.read_csv('dealtr_name.csv')
    data_te = pd.read_csv('dealte_name.csv')

    data_tr['Title1'] = data_tr["Title"].map(lambda x:silce_title1(x))
    data_te['Title1'] = data_te["Title"].map(lambda x: silce_title1(x))


    """
    data_tr['Mr'] = data_tr['Title'].map(lambda x:silce_title(x,'Mr'))
    data_tr['Mrs'] = data_tr['Title'].map(lambda x: silce_title(x, 'Mrs'))
    data_tr['Miss'] = data_tr['Title'].map(lambda x: silce_title(x, 'Miss'))
    data_tr['Master'] = data_tr['Title'].map(lambda x: silce_title(x, 'Master'))
    data_te['Mr'] = data_te['Title'].map(lambda x: silce_title(x, 'Mr'))
    data_te['Mrs'] = data_te['Title'].map(lambda x: silce_title(x, 'Mrs'))
    data_te['Miss'] = data_te['Title'].map(lambda x: silce_title(x, 'Miss'))
    data_te['Master'] = data_te['Title'].map(lambda x: silce_title(x, 'Master'))
    """
    data_tr.to_csv('midtr.csv', index=False)
    data_te.to_csv('midte.csv', index=False)