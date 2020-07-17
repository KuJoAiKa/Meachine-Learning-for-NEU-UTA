import numpy as np
from scipy import *
from scipy.optimize import fmin_bfgs
import matplotlib.pyplot as plt


Dataset = []
Labels = []
y = []

#170 57 32 190 95 28 150 45 35 168 65 29 175 78 26 185 90 32 171 65 28 155 48 31 165 60 27 182 80 30 175 69 28 178 80 27 160 50 31 170 72 30
#W M W M M M W W W M W M W M
#162 53 28 168 75 32 175 70 30 180 85 29

def inputDataset():
    strDataset = input("Input all Data points splitted in space:")
    Dataset = strDataset.split(" ")
    Dataset = list(map(int, Dataset))
    Dataset = np.array(Dataset)
    Dataset = Dataset.reshape(-1, 3)
    return Dataset

def inputLabels(Dataset,y):
    strlabels = input("Input all labels:")
    Labels = strlabels.split(" ")
    if Dataset.shape[0] != len(Labels):
        print("Amounts of Dataset and labels are different")
        exit(0)

    for k in range(len(Labels)):
        if Labels[k] == 'W':
            y.append(0)
        if Labels[k] == 'M':
            y.append(1)

    #output the whole Dataset
    listDataset = Dataset.tolist()
    temp = list(zip(listDataset, Labels, y))
    print("The whole Dataset:")
    for i in range(len(temp)):
        print(temp[i],end="\n")

    y = np.array(y)
    return Labels,y

def sigmoid(z):
    h = np.zeros((len(z), 1))
    h=1.0/(1.0+np.exp(-z))
    return h

def costFunction(initial_theta, X, y,inital_lambda):
    m=len(y)
    J = 0
    h = sigmoid(np.dot(X, initial_theta))
    thetal = initial_theta.copy()
    thetal[0] = 0
    temp = np.dot(np.transpose(thetal), thetal)
    J = (-np.dot(np.transpose(y), np.log(h))-np.dot(np.transpose(1-y), np.log(1-h))+temp*inital_lambda/2)/m
    return J

def gradient(initial_theta, X, y, initial_lambda):
    m = len(y)
    grad = np.zeros((initial_theta.shape[0]))
    h = sigmoid(np.dot(X, initial_theta))
    thetal = initial_theta.copy()
    thetal[0] = 0
    grad = np.dot(np.transpose(X), h-y)/m+initial_lambda/m*thetal
    return grad

def mapFeature_sub(out,X1,X2):
    tempx1 = X1 ** 2
    tempx2 = X2 ** 2
    out = np.hstack([out, np.multiply(tempx1, X2).reshape(-1, 1), np.multiply(tempx2, X1).reshape(-1, 1),
                     np.multiply(X2, X1).reshape(-1, 1)])
    return out

def mapFeature(X1,X2,X3):
    degree = 1
    out = np.ones((X1.shape[0], 1))
    out = np.hstack([out, X1.reshape(-1, 1), X2.reshape(-1, 1), X3.reshape(-1, 1)]) #1,x1,x2,x3
    # out = np.hstack([out, (X1 ** 3).reshape(-1, 1), (X1 ** 3).reshape(-1, 1), (X1 ** 3).reshape(-1, 1)])#X1^3,X2^3,X3^3
    # out = np.hstack([out, (X1 ** 2).reshape(-1, 1), (X1 ** 2).reshape(-1, 1), (X1 ** 2).reshape(-1, 1)])#x1^2,x2^2,x2^2
    # out = mapFeature_sub(out, X1, X2) #x1x2,x1^2 x2,x1 x2^2
    # out = mapFeature_sub(out, X2, X3)
    # out = mapFeature_sub(out, X1, X3)
    return out

def predict(X, theta):
    m = X.shape[0]
    p = np.zeros((m,1))
    p = sigmoid(np.dot(X, theta))
    for i in range (m):
        if p[i]>0.5:
            p[i] = 1
        else:
            p[i] = 0
    return p

if __name__ == '__main__':
    Dataset = inputDataset()
    Labels, y = inputLabels(Dataset,y)

    X = mapFeature(Dataset[:,0], Dataset[:,1], Dataset[:,2])

    #plot_data(Dataset, Labels)
    initial_theta = np.zeros((X.shape[1],1))
    initial_lambda = 0.0001

    J = costFunction(initial_theta,X,y,initial_lambda)
    #print(J)

    result = fmin_bfgs(costFunction, initial_theta, fprime=gradient, args=(X, y, initial_lambda))
    p = predict(X, result)
    print('acc on train:', np.mean(np.float64(p==y)*100))

    print("input test data")
    testData = inputDataset()
    testX = mapFeature(testData[:, 0], testData[:, 1], testData[:, 2])
    testp = predict(testX, result)
    testp = testp.tolist()
    for k in range(len(testp)):
        if testp[k] == 0:
            testp[k] ='W'
        if testp[k] == 1:
            testp[k] ='M'
    print('prediction', testp)

