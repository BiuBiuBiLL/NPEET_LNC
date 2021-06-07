from lnc import MI
from math import pi
import numpy as np
import numpy.random as nr

# Testing local non-uniform estimators

# total number of samples
N = 5000

# 2D Linear
noise = 1e-7
x = []
for i in range(N):
    x.append(nr.rand(1)[0])
y = []
y_for_ent = []
for i in range(N):
    y.append(x[i]+nr.rand(1)[0]*noise)
    y_for_ent.append([y[-1]])

# number of samples used for calculation
usedN = 500
print('Testing 2D linear relationship Y=X+Uniform_Noise')
print('noise level=' + str(noise) + ", Nsamples = " + str(usedN))
print('True MI(x:y)', MI.entropy(y_for_ent[:1000], k=1,base=np.exp(1),intens=0.0)-np.log(noise))
print('Kraskov MI(x:y)', MI.mi_Kraskov([x[:usedN],y[:usedN]],k=1,base=np.exp(1),intens=0.0))
print('LNC MI(x:y)', MI.mi_LNC([x[:usedN],y[:usedN]],k=5,base=np.exp(1),alpha=0.25,intens=0.0))

# 2D Quadratic
noise = 1e-7
x = []
for i in range(N):
    x.append(nr.rand(1)[0])
y = []
y_for_ent = []

for i in range(N):
    y.append(x[i]*x[i]+nr.rand(1)[0]*noise)
    y_for_ent.append([y[-1]])

# number of samples used for calculation
usedN = 1000
print('Testing 2D quadratic relationship Y=X^2+Uniform_Noise')
print('noise level=' + str(noise) + ", Nsamples = " + str(usedN))
print('True MI(x:y)', MI.entropy(y_for_ent[:1000],k=1,base=np.exp(1),intens=0.0)-np.log(noise))
print('Kraskov MI(x:y)', MI.mi_Kraskov([x[:usedN],y[:usedN]],k=1,base=np.exp(1),intens=0.0))
print('LNC MI(x:y)', MI.mi_LNC([x[:usedN],y[:usedN]],k=5,base=np.exp(1),alpha=0.25,intens=0.0))

#3D Linear
noise = 1e-7
x = []
for i in range(N):
    x.append(nr.rand(1)[0])
y = []
z = []
y_for_ent = []
z_for_ent = []
for i in range(N):
    y.append(x[i]+nr.rand(1)[0]*noise)
    z.append(x[i]+nr.rand(1)[0]*noise)
    y_for_ent.append([y[-1]])
    z_for_ent.append([z[-1]])

# number of samples used for calculation
usedN = 500
print('Testing 3D linear relationship Y=X+Uniform_Noise, Z=X+Uniform_Noise')
print('noise level=' + str(noise) + ", Nsamples = " + str(usedN))
print('True MI(x:y:z)', MI.entropy(y_for_ent[:1000],k=1,base=np.exp(1),intens=0.0)-np.log(noise)+MI.entropy(z_for_ent[:1000],k=1,base=np.exp(1),intens=0.0)-np.log(noise))
print('Kraskov MI(x:y:z)', MI.mi_Kraskov([x[:usedN],y[:usedN],z[:usedN]],k=1,base=np.exp(1),intens=0.0))
print('LNC MI(x:y:z)', MI.mi_LNC([x[:usedN],y[:usedN],z[:usedN]],k=5,base=np.exp(1),alpha=0.167,intens=0.0))


# 3D Quadratic
noise = 1e-7
x = []
for i in range(N):
    x.append(nr.rand(1)[0])
y = []
z = []
y_for_ent = []
z_for_ent = []
for i in range(N):
    y.append(x[i]*x[i]+nr.rand(1)[0]*noise)
    z.append(x[i]*x[i]+nr.rand(1)[0]*noise)
    y_for_ent.append([y[-1]])
    z_for_ent.append([z[-1]])

# number of samples used for calculation
usedN = 500
print('Testing 3D quadratic relationship Y=X^2+Uniform_Noise, Z=X^2+Uniform_Noise')
print('noise level=' + str(noise) + ", Nsamples = " + str(usedN))
print('True MI(x:y:z)', MI.entropy(y_for_ent[:1000],k=1,base=np.exp(1),intens=0.0)-np.log(noise)+MI.entropy(z_for_ent[:1000],k=1,base=np.exp(1),intens=0.0)-np.log(noise))
print('Kraskov MI(x:y:z)', MI.mi_Kraskov([x[:usedN],y[:usedN],z[:usedN]],k=1,base=np.exp(1),intens=0.0))
print('LNC MI(x:y:z)', MI.mi_LNC([x[:usedN],y[:usedN],z[:usedN]],k=5,base=np.exp(1),alpha=0.167,intens=0.0))
