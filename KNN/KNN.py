from numpy import *

def inputDataset():
    strDataset = input("Input all Data points splitted in space:")
    Dataset = strDataset.split(" ")
    Dataset = list(map(int, Dataset))
    Dataset = array(Dataset)
    Dataset = Dataset.reshape(-1, 3)
    return Dataset

def inputLabels(Dataset):
    strlabels = input("Input all labels:")
    Labels = strlabels.split(" ")
    if Dataset.shape[0] != len(Labels):
        print("Amounts of Dataset and labels are different")
        exit(0)

    #output the whole Dataset
    listDataset = Dataset.tolist()
    temp = list(zip(listDataset, Labels))
    print("The whole Dataset:")
    for i in range(len(temp)):
        print(temp[i],end="\n")
    return Labels

def KNNclassify(Target, Dataset, Labels, K):
    Num_Dataset = Dataset.shape[0]  #the amount of data points in dataset: int n
    Diff = tile(Target, (Num_Dataset,1)) - Dataset  #difference between input and data points: [n,3]
    squaredDiff = Diff ** 2
    sumDiff = sum(squaredDiff , axis = 1) #sum the diff in rows: [n]
    Distance = sumDiff ** 0.5  # distance

    ClassCount = {}
    sortDist = argsort(Distance) #get the index in ascending order

    for i in range(K):
        Label = Labels[sortDist[i]] #get the ith smallest label
        ClassCount[Label] = ClassCount.get(Label,0) + 1 #the amount + 1 ,default 0

    MaxCount = 0    #Find the most labels
    for Class, Amount in ClassCount.items():
        if Amount > MaxCount:
            MaxCount = Amount
            MaxClass = Class
    print("Votes of labels:")
    print(ClassCount)
    print("Classification:")
    print(MaxClass)
    return Distance, ClassCount, MaxClass

if __name__ == '__main__':
    Dataset = inputDataset()
    Labels = inputLabels(Dataset)

    K = int(input("Input K:"))
    if K > Dataset.shape[0]:
        print("K is out of range")
        exit(0)

    strTarget = input("Input the points to predict:")
    Target = strTarget.split(" ")
    Target = list(map(int, Target))
    Target = array(Target)

    KNNclassify(Target, Dataset, Labels, K)


