import pandas as pd #数据分析
import numpy as np #科学计算
from pandas import Series,DataFrame

if __name__=='__main__':
    data_train = pd.read_csv('alldat_tr.csv')
    data_test = pd.read_csv('alldat_te.csv')

    data_train.drop(['Name','Sex'],1,inplace=True)
    data_test.drop(['Name','Sex'],1,inplace=True)

    pd.set_option('display.max_row', 10)
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 1000)

    print(data_test)

    data_train.to_csv('dealtr_col1.csv',index=False)
    data_test.to_csv('dealte_col1.csv',index=False)