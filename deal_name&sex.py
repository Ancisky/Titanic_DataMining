import pandas as pd #数据分析
import numpy as np #科学计算
from pandas import Series,DataFrame

def replace_titles(x):
    title1 = x['Title']
    if title1 in ['Mr', 'Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
        x['Title']='Mr'
    elif title1 in ['Master']:
        x['Title']='Mr'
    elif title1 in ['Countess', 'Mme', 'Mrs']:
        x['Title']='Mrs'
    elif title1 in ['Mlle', 'Ms', 'Miss']:
        x['Title']='Miss'
    elif title1 == 'Dr':
        if x['Sex'] == 'Male':
            x['Title']='Mr'
        else:
            x['Title']='Mrs'
    elif title1 == '':
        if x['Sex'] == 'Male':
            x['Title']='Master'
        else:
            x['Title']='Miss'

def find_subname(name,subnames):#将完整的名字和所有的称谓的集合作为输入
    for subname in subnames:
        if name.find(subname) != -1:
            return subname#将匹配到的称谓返回
    return np.nan

def deal_name(temp):
    title_list = ['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                  'Dr', 'Ms', 'Mlle', 'Col', 'Capt', 'Mme', 'Countess',
                  'Don', 'Jonkheer']
    #所有称谓的集合
    temp['Title']=temp['Name'].map(lambda x:find_subname(x,title_list))
    #将整理出的称谓单独创造一个属性，命名为‘Title’
    for index1 in temp.index:
        title1 = temp.loc[index1,'Title']
        if title1 in ['Mr', 'Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
            temp.loc[index1,'Title'] = 'Mr'
        elif title1 in ['Countess', 'Mme', 'Mrs']:
            temp.loc[index1,'Title'] = 'Mrs'
        elif title1 in ['Mlle', 'Ms', 'Miss']:
            temp.loc[index1,'Title'] = 'Miss'
        elif title1 == 'Dr':
            if temp.loc[index1,'Sex'] == 'Male':
                temp.loc[index1,'Title'] = 'Mr'
            else:
                temp.loc[index1,'Title'] = 'Mrs'
        elif title1 == '':
            if temp.loc[index1,'Sex'] == 'Male':
                temp.loc[index1,'Title'] = 'Master'
            else:
                temp.loc[index1,'Title'] = 'Miss'
    #将所有称谓都转化为四个主要的称谓：‘Mr’，‘Mrs’，‘Miss’，‘Master’
if __name__ == '__main__':
    data_train=pd.read_csv('train2.csv')
    data_test=pd.read_csv('test2.csv')
    deal_name(data_train)
    deal_name(data_test)

    data_train.to_csv('dealtr_name.csv',index=False)
    data_test.to_csv('dealte_name.csv',index=False)


"""
    pd.set_option('display.max_row',10)
    pd.set_option('display.max_columns',20)
    pd.set_option('display.width',1000)
    print(data_train.describe())
    print(data_test.describe())
    # 显示数值属性统计结果
"""