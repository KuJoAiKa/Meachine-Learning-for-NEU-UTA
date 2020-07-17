import numpy as np
x = []
y = []
z = []
z1 = []
z2 = []
z3 = []
z4 = []
z1_err = []
z2_err = []
z3_err = []
z4_err = []

with open('PolyTest.txt','r')as train_data:
    while True:
        lines = train_data.readline()
        if not lines:
            break
        x_tmp, y_tmp,z_tmp = [float(i) for i in lines.split()]
        x.append(x_tmp)
        y.append(y_tmp)
        z.append(z_tmp)
print('Test Data:',x,'\n',y,'\n',z)


for k in range(len(x)):
    z1.append(3.12416858*x[k]+3.05840021*y[k]+1.74276999)
    z2.append(4.41437420+1.97318230*x[k]+2.95898476*y[k]+0.0970449459*(x[k]**2)+0.00100737883*(y[k]**2)+\
              0.0185376842*x[k]*y[k])
    z3.append(4.07188410+2.23298853*x[k]+2.95018777*y[k]+0.0479557096*(x[k]**2)+0.00888774447*(y[k]**2)+\
              0.00783664971*x[k]*y[k]+0.00276276244*(x[k]**3)+(-0.000609342905*(y[k]**3))+0.000726392379*((x[k]**2)*y[k])+\
              0.000257908443*(x[k]*y[k]**2))
    z4.append(3.98098213+(2.53083682*x[k])+2.72066789*y[k]+-0.0763670969*(x[k]**2)+0.0578769809*(y[k]**2)+\
              0.0666265720*x[k]*y[k]+0.0184043621*(x[k]**3)+-0.00231655951*(y[k]**3)+0.00466156618*(x[k]**2*y[k])+\
              (-0.0161850136*(x[k]*y[k]**2))+(-0.000591121669*(x[k]**4))+(-0.0000638981691*(y[k]**4))+\
              (-0.000679463719*(x[k]**3*y[k]))+0.000653496388*(x[k]*y[k]**3)+0.000667297660*(x[k]**2)*(y[k]**2))
    z1_err.append(z1[k] - z[k])
    z2_err.append(z2[k] - z[k])
    z3_err.append(z3[k] - z[k])
    z4_err.append(z4[k] - z[k])

z1_err = np.array(z1_err)
z2_err = np.array(z2_err)
z3_err = np.array(z3_err)
z4_err = np.array(z4_err)

print('Prediction:')
print(np.round(z1,4),'\n',np.round(z2,4),'\n',np.round(z3,4),'\n',np.round(z4,4))

print('\n Error(Prediction - Real):')
z1_err_4 = np.round(z1_err,4)
z2_err_4 = np.round(z2_err,4)
z3_err_4 = np.round(z3_err,4)
z4_err_4 = np.round(z4_err,4)

print(z1_err_4,'\n',z2_err_4,'\n',z3_err_4,'\n',z4_err_4)

print('Mean、Var、Std:')
print('1-order-pred:',np.mean(z1_err),np.var(z1_err),np.std(z1_err))
print('2-order-pred:',np.mean(z2_err),np.var(z2_err),np.std(z2_err))
print('3-order-pred:',np.mean(z3_err),np.var(z3_err),np.std(z3_err))
print('4-order-pred:',np.mean(z4_err),np.var(z4_err),np.std(z4_err))

np.round(np.mean(z1_err),4)