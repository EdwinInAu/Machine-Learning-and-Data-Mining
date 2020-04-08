#
# COMP9417
#
# Authors:
# Chongshi Wang
#

import numpy
import csv
import math
import matplotlib.pyplot as py
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

file_path = r'/Users/edwin/downloads/CreditCards.csv'

# read csv file
def read_function(file):
    data = []
    with open(file) as f:
        read_file = csv.reader(f)
        for line in read_file:
            data.append(line)
    final_data = numpy.array(data[1:]).astype(float)
    return final_data

# nomalization: dataset
def nomalization(data):
    length = len(data)
    for i in range(14):
        max_number = data[:,i].max()
        min_number = data[:,i].min()
        difference = max_number - min_number
        for j in range(length):
            data[j,i] = (data[j,i] - min_number)/difference
    return data

data = read_function(file_path)
nomalization_data = nomalization(data)
train_data = nomalization_data[:621,:]
test_data = nomalization_data[621:,:]
train_x = train_data[:,:13]
train_y = train_data[:,14]
test_x = test_data[:,:13]
test_y = test_data[:,14]

###################### Part A ########################

knn_classifier1 = KNeighborsClassifier(n_neighbors=2)
knn_classifier1.fit(train_x,train_y)
#pre1 = knn_classifier1.predict(train_x)
acc_score1 = knn_classifier1.score(train_x,train_y)
print("n_neighbors = 2 the accuracy score for training dataset:",acc_score1)
#pre2 = knn_classifier1.predict(test_x)
acc_score2 = knn_classifier1.score(test_x,test_y)
print("n_neighbors = 2 the accuracy score for test dataset:",acc_score2)

###################### Part B & Part C##########################

auc_index = []
auc_train = []
auc_test = []
max = 0

for i in range(1,31):
    knn_best = KNeighborsClassifier(n_neighbors= i)
    knn_best.fit(train_x,train_y)
    prediction_train = knn_best.predict_proba(train_x)
    prediction_test = knn_best.predict_proba(test_x)
    auc_t1 = roc_auc_score(y_true = train_y, y_score= prediction_train[:,1])
    auc_t2 = roc_auc_score(y_true= test_y, y_score= prediction_test[:,1])
    auc_train.append(auc_t1)
    auc_test.append(auc_t2)
    if auc_t2 > max:
        max = auc_t2
        auc_index.append(i)

print("AUC score of the optimal number of neighbours is:",max)
print("the optimal number of neighbours is:",auc_index[-1])

py.plot(auc_train)
py.xlabel("the value of k")
py.ylabel("AUC score")
py.title("AUC score: train dataset")
py.show()

py.plot(auc_test)
py.xlabel("the value of k")
py.ylabel("AUC score")
py.title("AUC score: test dataset")
py.show()

###################### Part D##########################

# K = 2
precision_k0 = precision_score(y_true= test_y, y_pred= pre2)
recall_k0 = recall_score(y_true= test_y, y_pred= pre2)
print("when k = 2, the precision score is:",precision_k0)
print("when k = 2, the recall score is",recall_k0)

# K = 5
knn_5 = KNeighborsClassifier(n_neighbors=5)
knn_5.fit(train_x, train_y)
pre5 = knn_5.predict(test_x)
precision_k5 = precision_score(y_true= test_y, y_pred= pre5)
recall_k5 = recall_score(y_true= test_y, y_pred= pre5)
print("when k = 5, the precision score is",precision_k5)
print("when k = 5, the recall score is",recall_k5)





