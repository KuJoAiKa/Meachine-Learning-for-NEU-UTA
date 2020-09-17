import math


class Node:
    value = ""
    children = []
    def __init__(self,val,dictionary):
        self.setValue(val)
        self.genChildren(dictionary)

    def __str__(self):
        return str(self.value)

    def setValue(self,val):
        self.value = val

    def genChildren(self,dictionary):
        if (isinstance(dictionary, dict)):
            self.children = dictionary.keys()


def find(item,list):
    for i in list:
        if item(i):
            return True
        else:
            return False

def majority(attributes, data, target):
    valFreq = {}
    index = attributes.index(target)

    for tuple in data:
        if(valFreq.has_key(tuple[index])):
            valFreq[tuple[index]] += 1
        else:
            valFreq[tuple[index]] = 1

    max = 0
    major = ""
    for key in valFreq.keys():
        if valFreq[key]>max:
            max = valFreq[key]
            major = key
    return major

def entropy(attributes, data, targetAttr):
    valFreq = {}
    dataEntropy = 0.0

    i = 0
    for entry in attributes:
        if(targetAttr == entry):
            break
        ++i

    for entry in data:
        if(valFreq.has_key(entry[i])):
            valFreq[entry[i]] += 1.0
        else:
            valFreq[entry[i]] = 1.0

    for freq in valFreq.values():
        dataEntropy += (-freq/len(data)) * math.log(freq/len(data), 2)
    return dataEntropy

def gain(attributes,data,attr,targetAttr):
    valFreq = {}
    subsetEntropy = 0.
    i = attributes.index(attr)

    for entry in data:
        if(valFreq.has_key(entry[i])):
            valFreq[entry[i]] += 1.0
        else:
            valFreq[entry[i]] = 1.0

    for val in valFreq.keys():
        valProb = valFreq[val] / sum(valFreq.values())
        dataSubset = [entry for entry in data if entry[i] ==val ]
        subsetEntropy += valProb * entropy(attributes, dataSubset, targetAttr)

    return  (entropy(attributes, data, targetAttr) - subsetEntropy)

def chooseAttr(data,attributes,target):
    best = attributes[0]
    maxGain = 0;
    for attr in attributes:
        newGain = gain(attributes, data, attr, target)
        if newGain > maxGain:
            maxGain = newGain
            best = attr
    return best

def getValues(data, attributes, attr):
    index = attributes.index(attr)
    values = []
    for entry in data:
        if entry[index] not in values:
            values.append(entry[index])
    return values

def getExamples(data,attributes, best, val):
    examples = [[]]
    index = attributes.index(best)
    for entry in data:
        if(entry[index] == val):
            newEntry = []
            for i in range(0,len(entry)):
                if(i != index):
                    newEntry.append(entry[i])
            examples.append(newEntry)
    examples.remove([])
    return examples

def makeTree(data,attributes, target, recursion):
    recursion += 1
    data = data[:]
    vals = [record[attributes.index(target)]for record in data]
    default = majority(attributes, data, target)

    if not data or len(attributes) -1 <=0:
        return default
    elif vals.count(vals[0]) == len(vals):
        return vals[0]
    else:
        best = chooseAttr(data,attributes,target)
        tree = {best:{}}

        for val in getValues(data,attributes,best):
            examples = getExamples(data,attributes, best,val)
            newAttr = attributes[:]
            newAttr.remove(best)
            subtree = makeTree(examples, newAttr, target, recursion)
            tree[best][val] = subtree
    return tree