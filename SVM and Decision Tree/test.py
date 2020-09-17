import numpy as np
import math

# x1 = []
# x2 = []
# y = []
#
# with open('SVM_train.txt', 'r') as f:
#     while True:
#         lines = f.readline()
#         if not lines:
#             break
#         x1_tmp, x2_tmp, y_tmp = [float(i) for i in lines.split()]
#         x1.append(x1_tmp)
#         x2.append(x2_tmp)
#         y.append(y_tmp)
#     X = zip(x1, x2)
#     X = list(X)
#     X = np.array(X)
#     y = np.array(y)
#
#     pos = np.where(y == 1)
#     neg = np.where(y == 0)
#
#     print(pos,neg)
#     print(X[pos, 0], X[neg, 0], X[pos, 1],  X[neg, 1])
prop1 = 1/4
prop2 = 3/4
print(   -(prop1)*math.log(prop1, 2)-prop2*math.log(prop2, 2) )
print(0.72193+0.98523)