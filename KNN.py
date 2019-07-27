import pandas as pd #数据分析
import numpy as np #科学计算
from pandas import Series,DataFrame
from sklearn.neighbors import KNeighborsClassifier

if __name__ == '__main__':
    train_data = pd.read_csv('dealtr_final1.csv')#打开训练集
    test_data = pd.read_csv('dealte_final1.csv')#打开测试集
    result = pd.read_csv('gender_submission.csv')#真实的生还情况

    X = train_data[['Pclass', 'Age', 'Family','Fare','Cabin','Title1']]
    #将需要参与训练的属性加入到X中
    #X包括船票等级，年龄，家庭成员个数，船票价格，是否有房间号，称谓
    y = train_data['Survived']
    #分类标签为‘Survived'
    clf = KNeighborsClassifier(n_neighbors=10,weights='uniform',
                               algorithm='kd_tree',leaf_size=20,
                               p=1,metric='minkowski',metric_params=None,n_jobs=1)
    #参数：n_neighbors参与决策的最近邻居个数设置为10；
    #每个邻居投票所拥有权值，uniform表示权值相等，distance则表示按距离反比设定权值，此处为uniform
    #选用kd树搜索最邻近邻居，algorithm其他取值有'auto'自动，'ball_tree'球树搜索,'brute'暴力搜索，在属性较少时kd_tree效率高
    #叶子节点包含实例个数最少为20，叶子节点过小时对导致噪声数据影响变大，过大会导致分类不全
    #metric和p是连为一体的，minkowski算法是一种混合距离算法，p=1时选用的是曼哈顿距离算法,p=2时选择的是欧氏距离算法，此处选择p=1
    clf.fit(X, y)
    #将X和y放入训练器训练，可以建立分类模型

    te = test_data[['Pclass', 'Age', 'Family','Fare','Cabin', 'Title1']]#测试集选取同样的属性集合
    te = te.values
    indexs = 0
    for index in te:
        test_data.loc[indexs, 'Survived'] = clf.predict([index])[0]#使用模型进行预测
        indexs += 1
    print(test_data['Survived'])

    result['Survived1'] = test_data['Survived'].map(lambda x: x)
    print(result)

    rightnum = 0
    for indexs in result.index:
        if result.loc[indexs, 'Survived'] == result.loc[indexs, 'Survived1']:
            rightnum += 1

    matrix = [[0, 0], [0, 0]]
    for indexs in result.index:
        if result.loc[indexs, 'Survived'] == 0:
            if result.loc[indexs, 'Survived'] == result.loc[indexs, 'Survived1']:
                matrix[0][0] += 1
            else:
                matrix[1][0] += 1
        else:
            if result.loc[indexs, 'Survived'] == result.loc[indexs, 'Survived1']:
                matrix[1][1] += 1
            else:
                matrix[0][1] += 1
    print(clf)
    print('正确率:'+str(rightnum / 418) + '   正确分类个数:' + str(rightnum))
    print('True_unsur True_sur')
    print(str(matrix[0][0])+'        '+str(matrix[0][1])+'      predict_unsur')
    print(str(matrix[1][0])+'         '+str(matrix[1][1])+'      predict_sur')
    TP = matrix[1][1]
    FN = matrix[0][1]
    FP = matrix[1][0]
    TN = matrix[0][0]
    print("生还，预测生还"+str(TP))
    print('生还，预测死亡'+str(FN))
    print('死亡，预测生还'+str(FP))
    print('死亡，预测死亡'+str(TN))
    print('AccuracyRate(准确率):'+str((TP+TN)/(TP+TN+FN+FP)))
    print('ErrorRate(误分率:)'+str((FN+FP)/(TP+TN+FN+FP)))
    print('Recall(召回率):'+str(TP/(TP+FN)))
    print('Precision(查准率):'+str(TP/(TP+FP)))
    print('False Positive Rate(错误接收率):'+str(FP/(FP+TN)))
    print('FalseRejection Rate(错误拒绝率):'+str(FN/(TP+FN)))
    """
    for indexs in test_data.index:
        np.int64(test_data.loc[indexs,'Survived'])
    test_data[['PassengerId','Survived']].to_csv('Submission.csv',index=False)
    """


