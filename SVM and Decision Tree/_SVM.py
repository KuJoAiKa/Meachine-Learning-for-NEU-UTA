import numpy as np
from sklearn import svm
from matplotlib import  pyplot as plt

x1 = []
x2 = []
y = []

def plot_data(X,y): #data points
    plt.figure(figsize=(8,8))
    pos = np.where(y == 1)
    neg = np.where(y == -1)
    p1, = plt.plot(np.ravel(X[pos, 0]), np.ravel(X[pos, 1]),'ro')
    p2, = plt.plot(np.ravel(X[neg, 0]), np.ravel(X[neg, 1]),'g^')
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend([p1, p2],["y==1","y==-1"])
    return plt

def plot_decisionBoundary(X, y , model): #decision boundary
    plt = plot_data(X,y)
    w = model.coef_

    b = model.intercept_
    xp = np.linspace(0, 5, 100)
    yp = -(w[0,0] * xp +b ) / w[0,1]

    plt.xlim((0, 5))
    plt.ylim((0, 5))
    plt.plot(xp, yp, 'b-')


    SV_index = model.support_    #margin
    k = -w[0][0] / w[0][1]
    for j in SV_index:
        b_ = X[j][1] - k * X[j][0]
        plt.scatter(X[j][0], X[j][1], s=100 , c='', alpha=0.5, linewidth=1.0, edgecolor='red')
        #plt.plot(X[j][0], w * X[j][0] + b_, )
        x = np.arange(0, 5, 0.01)
        y = k * x + b_
        if y[j] == 1:
            plt.scatter(x, y, s=5, linewidths=0.3, marker='.')
        else:
            plt.scatter(x, y, s=5, linewidths=0.3, marker='.')
    plt.show()

if __name__ == "__main__":
    with open('SVM_train.txt', 'r') as f:
        while True:
            lines = f.readline()
            if not lines:
                break
            x1_tmp, x2_tmp, y_tmp = [float(i) for i in lines.split()]
            x1.append(x1_tmp)
            x2.append(x2_tmp)
            y.append(y_tmp)
        X = zip(x1,x2)
        X = list(X)
        X = np.array(X)
        y = np.array(y)

    model = svm.SVC(C=100, kernel='linear', shrinking=True).fit(X,y)
    #plot_decisionBoundary(X,y,model)
    print(model.support_vectors_)