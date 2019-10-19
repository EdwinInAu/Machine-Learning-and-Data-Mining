//
// COMP9417
//
// Authors:
// Chongshi Wang
//

import csv
import math
import numpy
import matplotlib.pyplot as py

file_path = r'/Users/edwin/downloads/Advertising.csv'

def readFunction(file):
    data = []
    with open(file) as f:
        readFile = csv.reader(f)
        for line in readFile:
            data.append(line)
    final_data = numpy.array(data[1:]).astype(float)
    return final_data

def nomalisation(output_data):
    for i in range(1,4):
        max_number = output_data[:,i].max()
        min_number = output_data[:,i].min()
        difference = max_number - min_number
        for j in range(200):
            output_data[j,i] =  (output_data[j,i] - min_number)/difference
    return output_data

def gradient_descent(train_data,theta_0,theta_1,learning_rate,attribute):
    true_y = train_data[:,4]
    train_x = train_data[:,attribute]
    cost = []
    for i in range(500):
        train_y = theta_0 + theta_1 * train_x
        error = true_y - train_y
        cost_function = numpy.mean((error * error))
        cost_theta_0 = numpy.mean(error)
        cost_theta_1 = numpy.mean(error * train_x)
        cost.append(cost_function)
        theta_0 = theta_0 + learning_rate * cost_theta_0
        theta_1 = theta_1 + learning_rate * cost_theta_1
    return theta_0,theta_1,cost

def test(theta_0,theta_1,attribute,test_data):
    test_x = test_data[:,attribute]
    test_y = test_data[:,4]
    prediction_y = theta_0 + theta_1 * test_x
    rmse = math.sqrt(numpy.mean((test_y - prediction_y) * (test_y - prediction_y)))
    return rmse

output_data = readFunction(file_path)
nomalisation_data = nomalisation(output_data)
train_data = nomalisation_data[:190,:]
test_data = nomalisation_data[190:,:]
learning_rate = 0.01

attribute1_theta_0, attribute1_theta_1, cost = gradient_descent(train_data,-1,-0.5,learning_rate,1)
print("TV theta0: ", attribute1_theta_0,"TV theta1: ",attribute1_theta_1)

py.plot(cost)
py.xlabel("Iteration")
py.ylabel("Cost")
py.show()

attribute2_theta_0, attribute2_theta_1, cost = gradient_descent(train_data,-1,-0.5,learning_rate,2)
print("Radio theta0: ", attribute2_theta_0,"Radio theta1: ",attribute2_theta_1)

attribute3_theta_0, attribute3_theta_1, cost = gradient_descent(train_data,-1,-0.5,learning_rate,3)
print("Newspaper theta0: ", attribute3_theta_0,"Newspaper theta1: ",attribute3_theta_1)

rmse0 = test(attribute1_theta_0,attribute1_theta_1,1,train_data)
print("RMSE TV train: ",rmse0)

rmse1 = test(attribute1_theta_0,attribute1_theta_1,1,test_data)
print("RMSE TV test: ",rmse1)

rmse2 = test(attribute2_theta_0,attribute2_theta_1,2,test_data)
print("RMSE Radio test: ", rmse2)

rmse3 = test(attribute3_theta_0,attribute3_theta_1,3,test_data)
print("RMSE Newspaper test: ", rmse3)