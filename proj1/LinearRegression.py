import numpy as np
from numpy.linalg import inv
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

datasets =[]
x_array = []
x_array1 = []
x_array2 = []
y_array = []

with open('PolyTrain.txt','r')as train_data:
    while True:
        lines = train_data.readline()
        if not lines:
            break
        x1_tmp, x2_tmp,y_tmp = [float(i) for i in lines.split()]
        x_array1.append(x1_tmp)
        x_array2.append(x2_tmp)
        y_array.append(y_tmp)
    x_array1 = np.array(x_array1)
    x_array2 = np.array(x_array2)
    y_array = np.array(y_array)

print(x_array1,'\n',x_array2,'\n',y_array)


m = len(x_array1)  # 方程个数
A = np.ones(m).reshape(m, 1)
A = np.hstack([A, x_array1.reshape(m, 1), x_array2.reshape(m, 1)])  # 1,x,y
items1 = ['1','x','y']
#X = solve(np.dot(A.T, A), np.dot(A.T, y_array.T))
X = np.dot(inv(np.dot(A.T, A)), np.dot(A.T, y_array.T))
X,items1 = X.reshape(-1,1), np.array(items1).reshape(-1,1)
X = np.append(X,items1,axis=1)
print("1-order:\n", X)


A = np.ones(m).reshape(m, 1)
for i in range(2):
    if i==0:
        A = np.hstack([A, x_array1.reshape(m, 1), x_array2.reshape(m, 1)]) #1,x,y
    if i==1:
        A = np.hstack([A, (x_array1 ** 2).reshape(m, 1), (x_array2 ** 2).reshape(m, 1),  #x^2, y^2
                       np.multiply(x_array1, x_array2).reshape(m,1)])  #xy
items2 = ['1','x','y','x^2','y^2','xy']
X = np.dot(inv(np.dot(A.T, A)), np.dot(A.T, y_array.T))
X,items2 = X.reshape(-1,1), np.array(items2).reshape(-1,1)
X = np.append(X,items2,axis=1)
print("2-order:\n", X)


A = np.ones(m).reshape(m, 1)
for i in range(3):
    tempx1 = x_array1 ** 2
    tempx2 = x_array2 ** 2
    if i==0:
        A = np.hstack([A, x_array1.reshape(m, 1), x_array2.reshape(m, 1)]) #1,x,y
    if i==1:
        A = np.hstack([A, (x_array1 ** 2).reshape(m, 1), (x_array2 ** 2).reshape(m, 1), #x^2, y^2,
                       np.multiply(x_array1, x_array2).reshape(m, 1)])  #xy
    if i==2:
        A = np.hstack([A, (x_array1 ** 3).reshape(m, 1), (x_array2 ** 3).reshape(m, 1), #x^3,y^3
                       np.multiply(tempx1, x_array2).reshape(m, 1), np.multiply(x_array1, tempx2).reshape(m, 1)]) #x^2y, xy^2
items3 = ['1','x','y','x^2','y^2','xy','x^3','y^3','yx^2','xy^2']
X = np.dot(inv(np.dot(A.T, A)), np.dot(A.T, y_array.T))
X,items3 = X.reshape(-1,1), np.array(items3).reshape(-1,1)
X = np.append(X,items3,axis=1)
print("3-order:\n", X)


A = np.ones(m).reshape(m, 1)
for i in range(4):
    tempx1 = x_array1 ** 2
    tempx2 = x_array2 ** 2
    tempx13 = x_array1 ** 3
    tempx23 = x_array2 ** 3
    if i==0:
        A = np.hstack([A, (x_array1 ** 1).reshape(m, 1), (x_array2 ** 1).reshape(m, 1)])#1,x,y
    if i==1:
        A = np.hstack([A, (x_array1 ** 2).reshape(m, 1), (x_array2 ** 2).reshape(m, 1),#x^2,y^2
                       np.multiply(x_array1, x_array2).reshape(m, 1)]) #xy
    if i==2:
        A = np.hstack([A, (x_array1 ** 3).reshape(m, 1), (x_array2 ** 3).reshape(m, 1), #x^3,y^3
                       np.multiply(tempx1, x_array2).reshape(m, 1), np.multiply(x_array1, tempx2).reshape(m, 1)]) #x^2y, xy^2
    if i==3:
        A = np.hstack([A, (x_array1 ** 4).reshape(m, 1), (x_array2 ** 4).reshape(m, 1),  #x^4, y^4
                       np.multiply(tempx13, x_array2).reshape(m, 1), np.multiply(x_array1, tempx23).reshape(m, 1),#x^3y, y^3x
                       np.multiply(tempx1, tempx2).reshape(m, 1)]) #x^2y^2
items4 = ['1','x','y','x^2','y^2','xy','x^3','y^3','yx^2','xy^2','x^4','y^4','yx^3','xy^3','x^2y^2']
X = np.dot(inv(np.dot(A.T, A)), np.dot(A.T, y_array.T))
X,items4 = X.reshape(-1,1), np.array(items4).reshape(-1,1)
X = np.append(X,items4,axis=1)
print("4-order:\n", X)

#=========================================================================================
#plot


fig = plt.figure()
ax = Axes3D(fig)               # to work in 3d

x_surf=np.arange(0, 10, 0.1)                # generate a mesh
y_surf=np.arange(0, 10, 0.1)
x_surf, y_surf = np.meshgrid(x_surf, y_surf)
z_surf1 = 3.12416858*x_surf+3.05840021*y_surf+1.74276999    # ex. function, which depends on x and y
z_surf2 = 4.41437420+1.97318230*x_surf+2.95898476*y_surf+0.0970449459*(x_surf**2)+0.00100737883*(y_surf**2)+\
          0.0185376842*x_surf*y_surf
z_surf3 = 4.07188410+2.23298853*x_surf+2.95018777*y_surf+0.0479557096*(x_surf**2)+0.00888774447*(y_surf**2)+\
          0.00783664971*x_surf*y_surf+0.00276276244*(x_surf**3)+(-0.000609342905*(y_surf**3))+0.000726392379*(x_surf**2*y_surf)+\
          0.000257908443*(x_surf*y_surf**2)
z_surf4 = 3.98098213+(2.53083682*x_surf)+2.72066789*y_surf+-0.0763670969*(x_surf**2)+0.0578769809*(y_surf**2)+\
          0.0666265720*x_surf*y_surf+0.0184043621*(x_surf**3)+-0.00231655951*(y_surf**3)+0.00466156618*(x_surf**2*y_surf)+\
          (-0.0161850136*(x_surf*y_surf**2))+(-0.000591121669*(x_surf**4))+(-0.0000638981691*(y_surf**4))+\
          (-0.000679463719*(x_surf**3*y_surf))+0.000653496388*(x_surf*y_surf**3)+0.000667297660*(x_surf**2)*(y_surf**2)
#ax.plot_surface(x_surf, y_surf, z_surf1, cmap=cm.hot);
#ax.plot_surface(x_surf, y_surf, z_surf2, cmap=cm.hot);
#ax.plot_surface(x_surf, y_surf, z_surf3, cmap=cm.hot);
ax.plot_surface(x_surf, y_surf, z_surf4, cmap=cm.hot);    # plot surface plot


x = x_array1.tolist()
y = x_array2.tolist()
z = y_array.tolist()
ax.scatter(x, y, z);                        # plot scatter plot

ax.set_xlabel('x label')
ax.set_ylabel('y label')
ax.set_zlabel('z label')

plt.show()