#循环拿出其中一个，与其他所有的比较
#K取1，3，5，
##循环拿出其中一个，与其他所有的比较
#输出正确标签与判断标签
#统计正确个数
#删除年龄列，再进行  print (np.delete(a,2,axis = 1)) 第3列

from KNN import *
from numpy import *


##循环拿出其中一个，与其他所有的比较

def KNNdistance(Target, Dataset):
    Num_Dataset = Dataset.shape[0]  # the amount of data points in dataset: int n
    Diff = tile(Target, (Num_Dataset, 1)) - Dataset  # difference between input and data points: [n,3]
    squaredDiff = Diff ** 2
    sumDiff = sum(squaredDiff, axis=1)  # sum the diff in rows: [n]
    Distance = sumDiff ** 0.5  # distance
    return Distance

if __name__ == '__main__':
    originDataset = inputDataset() #array
    attri2Dataset = delete(originDataset, 2, axis=1)
    Labels = inputLabels(attri2Dataset) #list

    for i in range(originDataset.shape[0]):
        Target = attri2Dataset[i]
        Target = array(Target)
        Distance = KNNdistance(Target, attri2Dataset)
        print(Target, '：')
        print("Distance to points:", Distance)