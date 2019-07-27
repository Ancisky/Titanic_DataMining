import pandas as pd #数据分析
import numpy as np #科学计算
from pandas import Series,DataFrame
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':

    train_data = pd.read_csv('dealtr_final1.csv')
    test_data = pd.read_csv('dealte_final1.csv')
    result  = pd.read_csv('gender_submission.csv')

    X = train_data[['Pclass', 'Age', 'Family', 'Fare','Cabin','Title1']]
    y = train_data['Survived']
    clf = RandomForestClassifier(n_estimators=150,criterion='gini',max_depth=None,
                                 min_samples_split=15,min_samples_leaf=25,min_weight_fraction_leaf=0,
                                 max_features=None,max_leaf_nodes=None,bootstrap=True,
                                 oob_score=True,n_jobs=-1,warm_start=False,class_weight=None)
    #n_estimators为决策树的个数，随机森林训练的方式是建立指定数量的决策树进行联合决策，
    #最后选取投票最多的选项进行决策，很大程度上降低了决策的偶然性，决策树数目越多，算法
    #正确率和决策稳定性聚会升高，但决策树过多带来的是算法的性能问题，消耗资源会变多，设置150
    #criterion选择最佳分裂属性的算法，有‘gini’（gini系数）和‘entropy’（信息增益），此处为gini
    #max_depth，设置树的最大深度，选择None，这样会使每个叶节点只有一个类别，或者到达最小样本数
    #min_samples_split，每个节点划分样本的最小数量，此处设置为15
    #min_samples_leaf，叶子节点的最少样本数
    #min_weight_fraction_leaf，叶子节点需要的最小权值，选择默认值0
    #max_features，选择最适属性时划分的特征不能超过该值，也就是在随机分配属性进行训练决
    # 策时，每次随机分配的属性数量不能超过此值，这里设置None，即样本所有特征总数
    #max_leaf_nodes,叶子树的最大的样本数，设置为None，即全部样本数
    #bootstrap是否有放回地抽样，选择True
    #oob_score，带外数据验证，在某次建立决策树训练时没有选中的数据就可以用来进行决策树的
    #模型进行验证，相较于交叉验证，这种验证对性能影响小，并且效果显著，选为True
    #warm_start,热启动，决定是否使用上次调用该类的结果然后增加新的,选择Fales
    #class_weight，属性标签的权重，设置为None
    clf.fit(X,y)

    te = test_data[['Pclass', 'Age', 'Family', 'Fare', 'Cabin', 'Title1']]
    te = te.values
    indexs = 0
    for index in te:
        test_data.loc[indexs, 'Survived'] = clf.predict([index])[0]
        indexs += 1

    result['Survived1'] = test_data['Survived'].map(lambda x: x)
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

    test_data[['PassengerId','Survived']].to_csv('Submission.csv',index=False)
#    for indexs in test_data.index:
#        np.int64(test_data.loc[indexs,'Survived'])
    #print(clf.feature_importances_)
    #test_data.add_prefix('Survived')
    #test_data['Survived'] = test_data[['Pclass', 'Age', 'Family', 'Fare', 'Embarked1','Cabin','Title1']].map(lambda x:clf.predict(x))

