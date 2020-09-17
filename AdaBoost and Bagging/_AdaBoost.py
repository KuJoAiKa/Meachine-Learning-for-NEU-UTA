from sklearn.linear_model import LogisticRegression
import numpy as np


def Input_data(filename):
    Variance = []
    Skewness = []
    Kurtosis = []
    Entropy = []
    y = []
    with open(filename, 'r')as data:
        while True:
            lines = data.readline()
            if not lines:
                break
            x1_tmp, x2_tmp, x3_tmp, x4_tmp, y_tmp = [float(i) for i in lines.split(',')]
            Variance.append(float(x1_tmp))
            Skewness.append(float(x2_tmp))
            Kurtosis.append(float(x3_tmp))
            Entropy.append(float(x4_tmp))
            if y_tmp == 0:
                y.append(-1)
            else:
                y.append(y_tmp)
        X = list(zip(Variance, Skewness, Kurtosis, Entropy))  # zip needs list()
        X = np.array(X)
        y = np.array(y)
    return X, y


def Evaluate(Pred, y_test):
    error_count = 0
    for i in range(len(Pred)):
        if Pred[i] != y_test[i]:
            error_count += 1
        else: continue
    error_rate = error_count/len(Pred)
    return error_rate


def AdaBoost(X_train, y_train, X_test, y_test, times):
    base = LogisticRegression(C=10,solver='lbfgs')
    models = []
    weights_models = []
    weights = []
    pred_eval = np.zeros([len(y_test)])

    for i in range(len(y_train)):
        weights.append(1/len(y_train))
    weights = np.array(weights)

    for i in range(times):
        models.append(base)
        models[i].fit(X_train, y_train, sample_weight=weights)
        pred_train = np.array(models[i].predict(X_train))
        pred_test = np.array(models[i].predict(X_test))
        #print("Error single:", Evaluate(pred_test,y_test))

        _e = 0
        for j in range(len(y_train)):
            if pred_train[j] != y_train[j]:
                _e += weights[j]
        #print(_e)
        if _e == 0 or _e >= 0.5:
            print(i+1,"models, Error count=0")

            break
        else:
            _alpha = np.log((1 - _e) / _e) * 0.5
            #print(_alpha)
            weights_models.append(_alpha)

        _z = np.multiply(weights,np.exp(-weights_models[i] * y_train * pred_train))
        weights = np.array(_z / np.sum(_z))


        pred_eval = [sum(x) for x in zip(pred_eval,[x * _alpha for x in pred_test])]   #the final clf
        temp = np.sign(pred_eval) #sign function
        print("Boost error:",Evaluate(temp,y_test))


if __name__ == '__main__':
    times_list = [10,25,50]
    X_train, y_train = Input_data('train.txt')
    X_test,y_test = Input_data('test.txt')

    ori_LR = LogisticRegression(C=10,solver='lbfgs').fit(X_train,y_train)  #a single
    single_Pred = ori_LR.predict(X_test)
    Error_rate = Evaluate(single_Pred, y_test)
    print("Error Rate with single LR:",Error_rate)

    for times in times_list:
        print("AdaBoost",times,"times:")
        AdaBoost(X_train,y_train,X_test,y_test,times)
