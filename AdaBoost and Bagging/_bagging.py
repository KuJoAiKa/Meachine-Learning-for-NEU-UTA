from sklearn.linear_model import LogisticRegression
import numpy as np
import random

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
            y.append(y_tmp)
        X = list(zip(Variance, Skewness, Kurtosis, Entropy))  # zip needs list()
        X = np.array(X)
        y = np.array(y)
    return X, y

def random_sample(X_train,y_train):
    len_data = X_train.shape[0]
    data = np.empty([len_data,4], dtype = float)
    label = np.empty([len_data], dtype = float)

    for i in range(len_data):
        temp_index = random.randint(0, len_data - 1)
        data[i] = X_train[temp_index]
        label[i] = y_train[temp_index]
    return data, label

def bagging_LR(X_train,y_train,X_test,y_test,times):
    Pred = np.zeros([times,len(X_test)],dtype=int)
    for i in range(times):
        X_train,y_train = random_sample(X_train,y_train)
        LR = LogisticRegression(solver='lbfgs').fit(X_train,y_train)
        Pred[i] = LR.predict(X_test)
        #print(LR.predict(X_test))
    return Pred

def prediction_vote(Pred): #pred [times , len_data]
    len_data = Pred.shape[1]
    Pred_new = []
    for i in range(len_data):
        n0 = list(Pred[:, i].T).count(0)
        n1 = list(Pred[:, i].T).count(1)
        if n0 > n1:
            Pred_new.append(0)
        else:
            Pred_new.append(1)
    Pred_vote = np.array(Pred_new)
    return Pred_vote

def Evaluate(Pred_vote, y_test):
    error_count = 0
    for i in range(len(Pred_vote)):
        if Pred_vote[i] != y_test[i]:
            error_count += 1
        else: continue
    error_rate = error_count/len(Pred_vote)
    return error_rate, error_count

if __name__ == '__main__':
    # stat = {}
    # mean = []
    # for k in range(10):
    #     times_list = [10,50,100]
    #     X_train, y_train = Input_data('train.txt')
    #     X_test,y_test = Input_data('test.txt')
    #
    #     ori_LR = LogisticRegression(solver='lbfgs').fit(X_train,y_train)
    #     single_Pred = ori_LR.predict(X_test)
    #     Error_rate,Error_count = Evaluate(single_Pred, y_test)
    #     print("Error Rate and count with single LR:",Error_rate,Error_count)
    #
    #     for times in times_list:
    #         Pred = bagging_LR(X_train,y_train,X_test,y_test,times)
    #         Pred_vote = prediction_vote(Pred)
    #         Error_rate,Error_count = Evaluate(Pred_vote, y_test)
    #         print("Bagging times: ",times,"\nError Rate: ",Error_rate,"\nError Count",Error_count)
    #         stat[times] = Error_count
    #     mean.append(stat)


    times_list = [10, 50, 100]
    X_train, y_train = Input_data('train.txt')
    X_test, y_test = Input_data('test.txt')

    ori_LR = LogisticRegression(solver='lbfgs').fit(X_train, y_train)
    single_Pred = ori_LR.predict(X_test)
    Error_rate, Error_count = Evaluate(single_Pred, y_test)
    print("Error Rate and count with single LR:", Error_rate, Error_count)

    for times in times_list:
        Pred = bagging_LR(X_train, y_train, X_test, y_test, times)
        Pred_vote = prediction_vote(Pred)
        Error_rate, Error_count = Evaluate(Pred_vote, y_test)
        print("Bagging times: ", times, "\nError Rate: ", Error_rate, "\nError Count", Error_count)