import pandas as pd #数据分析
import numpy as np #科学计算
from pandas import Series,DataFrame
from sklearn import svm

if __name__ == '__main__':
    train_data = pd.read_csv('dealtr_final1.csv')
    test_data = pd.read_csv('dealte_final1.csv')
    result = pd.read_csv('gender_submission.csv')

    X = train_data[['Pclass', 'Age', 'Family', 'Fare', 'Embarked1','Cabin','Title1']]
    y = train_data['Survived']
    clf = svm.SVC(C=0.3,kernel='poly',degree=1,gamma='auto',coef0=0,probability=False,shrinking=True,
                  tol=1e-3,max_iter=-1,random_state=None)
    #C：错误项的惩罚系数，在对训练集样本进行训练时，错误项惩罚系数越高，对分错样本的乘法程度越大，
    #也就会使模型在训练样本中的准确率升高，但同时导致的是训练出的模型泛化能力下降，对于噪声数据较多的
    #数据集来说错误项惩罚系数越高训练出的模型对测试数据效果越差，C值调小时模型对训练集精度没那么高但
    #泛化能力强，对测试集效果会有提升，此处设置C=0.3
    #kernel算法使用的核函数类型，此处选取‘poly’多项式核函数
    #degree只对多项式核函数游泳，指多项式核函数的阶数
    #gamma，核函数系数，此处设为‘auto’
    #coef0，核函数中的独立项，此处为默认值0
    #probability，选择不启用概率估计
    #shrinking采用启发式收缩方式
    #tol为svm停止训练的误差精度，设置为默认值1e^-3
    clf.fit(X,y)
    te = test_data[['Pclass', 'Age', 'Family', 'Fare', 'Embarked1','Cabin','Title1']]
    te = te.values
    indexs = 0
    for index in te:
        test_data.loc[indexs,'Survived'] = clf.predict([index])[0]
        indexs+=1
    result['Survived1'] = test_data['Survived'].map(lambda x:x)

    matrix = [[0,0],[0,0]]
    for indexs in result.index:
        if result.loc[indexs,'Survived']==0:
            if result.loc[indexs,'Survived'] == result.loc[indexs,'Survived1']:
                matrix[0][0]+=1
            else:
                matrix[1][0]+=1
        else:
            if result.loc[indexs,'Survived'] == result.loc[indexs,'Survived1']:
                matrix[1][1]+=1
            else:
                matrix[0][1]+=1
    rightnum = 0
    for indexs in result.index:
        if result.loc[indexs,'Survived'] == result.loc[indexs,'Survived1']:
            rightnum+=1

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
    #test_data.to_csv('final_predict.csv')
