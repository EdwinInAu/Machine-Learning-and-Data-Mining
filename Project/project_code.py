import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale,StandardScaler,MinMaxScaler

from sklearn import metrics, svm
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE, VarianceThreshold,SelectKBest,f_classif,chi2,SelectFromModel
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
import seaborn as sns

if __name__== '__main__':
    # X
    # miss2
    # summary
    data = pd.read_csv(f'summary.csv')
    np.set_printoptions(suppress=True)  # scientific notation to normal
    # 12
    # 15
    data_x = np.array(data.iloc[0:, :20])
    # data_x = np.array(data.iloc[0:, 1:20])
    x_available = []

    # Y
    # FLOURISHING
    # flourishing_result = pd.read_csv('flour.csv')
    # result_user = np.array(flourishing_result.iloc[:, 0])
    # data_y = flourishing_result.iloc[:, 1]

    # PANAS
    # 1 = pre_postive
    # 2 = pre_negative
    # 3 = post_postive
    # 4 = post_negative
    # 5 = pos_mean
    # 6 = neg_mean
    #
    panas_result = pd.read_csv('panas.csv')
    result_user = np.array(panas_result.iloc[:, 0])
     # change here
    data_y = panas_result.iloc[:, 6]

    mean_result = np.median(data_y)
    result_class = []
    for i in data_y:
        if i > mean_result:
            result_class.append('high')
        else:
            result_class.append('low')
    result_class = np.array(result_class)
    # print(len(result_class))
    for i in data_x:
        if i[0] in result_user:
            x_available.append(list(i[1:]))

    # X = minmax_scale(np.array(x_available))
    # X = np.array(x_available)
    X = MinMaxScaler().fit_transform(x_available)
    # Y = np.array(data_y.astype(int))
    # sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    # X = SelectKBest(chi2, k=8).fit_transform(X, result_class)
    # X = sel.fit_transform(x_available)
    # X = StandardScaler().fit_transform(x_available)


    # k_best = SelectKBest(score_func=f_classif, k=10)
    # best_fit = k_best.fit(x_available, result_class)
    # X = k_best.fit(x_available)
    '''
    __________________________________________________________________________
    Decomposition experiment:
        pca
        lda
    '''
    # pca
    # pca = PCA(n_components=2)
    # pcaX = pca.fit_transform(x_available)

    #lda
    # lda = LinearDiscriminantAnalysis(n_components=2)
    # lda.fit(X, Y)
    # newX = lda.transform(X)

    model_selection = {}
    model_selection2 =  {}


    # knn = KNeighborsRegressor(5, weights='uniform')
    # knn_model = knn.fit(newX, Y)
    # predict = knn_model.predict(newX)
    # score = np.mean(-cross_val_score(knn_model, newX, Y, cv=5, scoring='neg_mean_squared_error'))
    # print('knnregressor cross_val_score: ' + str(score))
    # print("knnregressor rmse: ", np.sqrt(metrics.mean_squared_error(Y, predict)))
    # print()
    # x_axis = [i for i in range(34)]
    # print(len(result_user), len(Y_train), len(predict))
    #
    # plt.scatter(x_axis, Y_train, c='b', label='prediction')
    # plt.scatter(x_axis, predict, c='g', label='prediction')
    # plt.show()

    # # dt regressor
    # dtr = DecisionTreeRegressor()
    # dtr.fit(X, Y)
    # dtr_pre = dtr.predict(X)
    # score = np.mean(-cross_val_score(dtr, X, Y, cv=5, scoring='neg_mean_squared_error'))
    # print('dtcregressor cross_val_score: ' + str(score))
    # print("dtcregressor rmse: ", np.sqrt(metrics.mean_squared_error(Y, dtr_pre)))
    # print()

    '''
    __________________________________________________________________________
    KnnClassifier:
    '''
    def knn_classifier(X_train, Y_train):
        model = KNeighborsClassifier()
        param_grid = {'n_neighbors': [1,2,3,4,5,6,7,8,9],
                      'weights': ['uniform', 'distance'],
                      }
        grid_search = GridSearchCV(model, param_grid, scoring='roc_auc')
        grid_search.fit(X_train, Y_train)
        best_parameters = grid_search.best_estimator_.get_params()
        # for para, val in list(best_parameters.items()):
        #     print(para, val)
        model = KNeighborsClassifier(n_neighbors=best_parameters['n_neighbors'],
                                     weights=best_parameters['weights'])
        # model.fit(X_train, Y_train)
        return model

    knnclass = knn_classifier(X, result_class)
    # knn_pre = knnclass.predict(X)
    model_selection2.setdefault('knn', knnclass)
    # score = np.mean(cross_val_score(knnclass, X, result_class, cv=10, scoring='roc_auc'))
    # print('knnclassifier cross_val_score: ' + str(score))
    # print("knnclassifier accuracy: ", metrics.accuracy_score(result_class, knn_pre))
    # print()


    '''
    __________________________________________________________________________
    SVM:
    '''

    def svm_cross_validation(X_train, Y_train):
        model = SVC(probability=True)
        param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
                      'gamma': [1,0.1,0.01, 0.001, 0.0001],
                      'kernel': ['rbf','linear']}
        grid_search = GridSearchCV(model, param_grid,scoring='roc_auc')
        grid_search.fit(X_train, Y_train)
        best_parameters = grid_search.best_estimator_.get_params()
        # for para, val in list(best_parameters.items()):
        #     print(para, val)
        model = SVC(kernel=best_parameters['kernel'], C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
        return model

    svm_model = svm_cross_validation(X, result_class)

    # svm_pre = svm_model.predict(X)
    model_selection2.setdefault('svm', svm_model)
    # print(metrics.classification_report(result_class, svm_pre))
    # print(svm_pre)
    # score = np.mean(cross_val_score(svm_model, X, result_class, cv=9, scoring='roc_auc'))
    # print('svm cross_val_score: ' + str(score))
    # print("svm accuracy: ", metrics.accuracy_score(result_class, svm_pre))
    # print()


    '''
    __________________________________________________________________________
    DTC:
    
    '''
    def DTC(X_train, Y_train):
        model = DecisionTreeClassifier(splitter='best',random_state=0,criterion='entropy')
        param_grid = {'max_depth': [1,2,3,4]}
        grid_search = GridSearchCV(model, param_grid, n_jobs=8, verbose=1, scoring='roc_auc')
        grid_search.fit(X_train, Y_train)
        best_parameters = grid_search.best_estimator_.get_params()
        # for para, val in list(best_parameters.items()):
        #     print(para, val)
        model = DecisionTreeClassifier(splitter='best',max_depth=best_parameters['max_depth'],random_state=0,criterion='entropy')
        # model.fit(X_train, Y_train)
        return model
    dtc = DTC(X,result_class)
    # dtc_pre = dtc.predict(X)
    model_selection.setdefault('dtc', dtc)
    # score = np.mean(cross_val_score(dtc, X, result_class, cv=9, scoring='roc_auc'))
    # print('dtcclassifier cross_val_score: ' + str(score))
    # print("dtc accuracy: ", metrics.accuracy_score(result_class, dtc_pre))
    # print()


    '''
    __________________________________________________________________________
    Bagging classifier:
    '''
    # def BG_classifier(X_train, Y_train):
    #     model = BaggingClassifier(KNeighborsClassifier(), random_state=0)
    #     param_grid = {'base_estimator': [KNeighborsClassifier()]}
    #     grid_search = GridSearchCV(model, param_grid, n_jobs=8, verbose=1, scoring='roc_auc')
    #     grid_search.fit(X_train, Y_train)
    #     best_parameters = grid_search.best_estimator_.get_params()
    #     # for para, val in list(best_parameters.items()):
    #     #     print(para, val)
    #     model = BaggingClassifier(random_state=0,base_estimator = best_parameters['base_estimator'])
    #     # model.fit(X_train, Y_train)
    #     return model
    #
    # bg = BG_classifier(X, result_class)
    # # bg_pre = bg.predict(X)
    # model_selection2.setdefault('BG', bg)
    # score = np.mean(cross_val_score(bg, X, result_class, cv=9, scoring='roc_auc'))
    # print('BGclassifier cross_val_score: ' + str(score))
    # print("BG accuracy: ", metrics.accuracy_score(result_class, bg_pre))
    # print()

    '''
    __________________________________________________________________________
    Randomforest:
    '''
    def Random_forest_classifier(X_train, Y_train):
        model = RandomForestClassifier(random_state =0)
        param_grid = {'n_estimators': [1,5,8,10,12,14],
                      'max_depth':[1,2,3,4]}
        grid_search = GridSearchCV(model, param_grid,scoring='roc_auc')
        grid_search.fit(X_train, Y_train)
        best_parameters = grid_search.best_estimator_.get_params()
        # for para, val in list(best_parameters.items()):
        #     print(para, val)
        model = RandomForestClassifier(random_state=0, n_estimators=best_parameters['n_estimators'],
                                       max_depth=best_parameters['max_depth'])
        # model.fit(X_train, Y_train)
        return model

    r_forest = Random_forest_classifier(X, result_class)
    # feature_importance = r_forest.feature_importances_
    # print(data.iloc[0, :])
    # print(feature_importance)
    # feature_importance = 100.0 * (feature_importance / feature_importance.max())
    # sorted_idx = np.argsort(feature_importance)
    # print(sorted_idx)

    # r_pre = r_forest.predict(X)
    model_selection.setdefault('randomforest', r_forest)
    # score = np.mean(cross_val_score(r_forest, X, result_class, cv=9, scoring='roc_auc'))
    # print('Randomforest classifier cross_val_score: ' + str(score))
    # print("Randomforest accuracy: ", metrics.accuracy_score(result_class, r_pre))
    # print()

    #
    '''
    __________________________________________________________________________
    MultinomialNB:
    '''
    MNB = MultinomialNB()
    # # MNB.fit(X, result_class)
    # # MNB_pre = MNB.predict(X)
    # # score = np.mean(cross_val_score(MNB, X, result_class, cv=10, scoring='roc_auc'))
    model_selection.setdefault('MNB', MNB)
    # print('MNB classifier cross_val_score: ' + str(score))
    # print("MNB accuracy: ", metrics.accuracy_score(result_class, MNB_pre))
    # print()


    # LinearRegression
    LR = LogisticRegression(solver = 'liblinear', random_state = 0)
    # new_lr = SelectFromModel(LR)
    # newX = new_lr.fit_transform(X, result_class)
    # score = np.mean(cross_val_score(LR, newX, result_class, cv=9, scoring='roc_auc'))
    # print('LR classifier cross_val_score: ' + str(score))
    # # print("LR accuracy: ", metrics.accuracy_score(result_class, LR.predict(newX)))
    # print()
    # LR.fit(X,result_class)
    # lr_pre = LR.predict(X)
    model_selection.setdefault('LR', LR)

    # feature selection

    '''
      __________________________________________________________________________
      GaussianNB:
      '''

    #
    # def GNB_SELECTION(X_train, Y_train):
    #     model = GaussianNB()
    #     param_grid = {'average': ['micro','macro', 'weighted','samples']}
    #     grid_search = GridSearchCV(model, param_grid, scoring='roc_auc')
    #     grid_search.fit(X_train, Y_train)
    #     best_parameters = grid_search.best_estimator_.get_params()
    #     # for para, val in list(best_parameters.items()):
    #     #     print(para, val)
    #     model = GaussianNB(average = )
    #     # model.fit(X_train, Y_train)
    #     return model

    GNB = GaussianNB()
    # GNB.fit(X, result_class)
    # GNB_pre = GNB.predict(X)

    model_selection2.setdefault('GNB', GNB)
    # score = np.mean(cross_val_score(GNB, X, result_class, cv=9, scoring='roc_auc'))
    # print('GNB classifier cross_val_score: ' + str(score))
    # print("GNB accuracy: ", metrics.accuracy_score(result_class, GNB_pre))
    # print()

    # AdaBoostClassifier
    # ADA = AdaBoostClassifier()
    # # ADA.fit(X, result_class)
    # # ADA_PRE = ADA.predict(X)
    # model_selection.setdefault('ADA', ADA)
    # score = np.mean(cross_val_score(ADA, X, result_class, cv=9, scoring='roc_auc'))
    # # print('ADA classifier cross_val_score: ' + str(score))
    # print("ADA accuracy: ", metrics.accuracy_score(result_class, ADA_PRE))
    # print()

    # # BernoulliNB
    # BNB = BernoulliNB()
    # # BNB.fit(X, result_class)
    # # BNB_pre = BNB.predict(X)
    # model_selection.setdefault('BNB', BNB)
    # score = np.mean(cross_val_score(BNB, X, result_class, cv=9, scoring='roc_auc'))
    # # print('BNB classifier cross_val_score: ' + str(score))
    # # print("BNB accuracy: ", metrics.accuracy_score(result_class, BNB_pre))
    # print()

    # cv
    score_rank = {}
    for model in model_selection.keys():

        # model_selection[model].fit(X, result_class)
        new_model = SelectFromModel(model_selection[model])
        newX = new_model.fit_transform(X, result_class)
        # pre = model_selection[model].predict(X)
        score = np.mean(cross_val_score(model_selection[model], newX, result_class, cv=9, scoring='roc_auc'))
        score_rank.setdefault(model,score)

    for model in model_selection2.keys():
        model_selection2[model].fit(X, result_class)
        pre = model_selection2[model].predict(X)
        score = np.mean(cross_val_score(model_selection2[model], X, result_class, cv=9, scoring='roc_auc'))
        score_rank.setdefault(model, score)

    score_rank = reversed(sorted(score_rank.items(), key=lambda x: x[1],))
    for i in score_rank:
        print(i)