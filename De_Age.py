import pandas as pd #数据分析
import numpy as np #科学计算
from pandas import Series,DataFrame

def de_age1(x):
    if x>=0 and x<=10:
        return (1)
    elif x>10 and x<=20:
        return (2)
    elif x>20 and x<=30:
        return (3)
    elif x>30 and x<=40:
        return (4)
    elif x>40 and x<=50:
        return (5)
    elif x>50 and x<=60:
        return (6)
    elif x>60 and x<=70:
        return (7)
    elif x>70 and x<=80:
        return (8)
    elif x>80 and x<=90:
        return (9)

def de_age2(x):
    if x>=0 and x<=15:
        return (1)
    elif x>15 and x<=25:
        return (2)
    elif x>25 and x<=40:
        return (3)
    elif x>40 and x<=60:
        return (4)
    elif x>60:
        return (5)


if __name__ == '__main__':
    train_data = pd.read_csv('dealtr_final.csv')
    test_data = pd.read_csv('dealte_final.csv')

    train_data['Age'] = train_data['Age'].map(lambda x:de_age2(x))
    test_data['Age'] = test_data['Age'].map(lambda x:de_age2(x))

    train_data.to_csv('dealtr_final1.csv',index=False)
    test_data.to_csv('dealte_final1.csv',index=False)