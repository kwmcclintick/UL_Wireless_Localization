import numpy as np
import scipy.io as sio
from os import listdir
from os.path import isfile, join
import math
import matplotlib.pyplot as plt
from scipy import stats


mypath = './NLOS_test_5_23_20/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
X = []
Y = []
vars = []
for file in onlyfiles:
    mat = sio.loadmat(mypath + file)
    rss_data = mat['file'][0][0][0][0][0][0]
    nlos_watt = rss_data[:,1]
    rss_data = 10*np.log10(rss_data[:, 1])
    idx = ~np.isnan(rss_data)
    rss_data = rss_data[idx]
    nlos_watt = nlos_watt[idx]
    vars.append(np.var(rss_data))
    # print(np.var(rss_data, axis=0))
    X.extend(rss_data)  # RSS values
    Y.extend(np.tile(mat['file'][0, 0][0][-1][0][-1][0], reps=(len(rss_data), 1)))  # tile stationary location labels
print('Estimated NLOS shadow fading variance: ',np.mean(vars))
X = np.array(X)
Y = np.array(Y)
rec_loc = np.array([0, 12.99])
dist = 10*np.log10(np.linalg.norm(rec_loc - Y, axis=1, ord=2))
slope, intercept, r_value, p_value, std_err = stats.linregress(dist,X)
print('Estimated NLOS alpha: ',slope)
print('Estimated P0: ', intercept)
plt.scatter(dist,X)
plt.xlabel('10Log10 Distance (m)')
plt.ylabel('Power (dBm)')
plt.title(r'NLOS $\alpha=$ %f $\sigma=$ %f $P0=$ %f' % (slope, np.mean(vars), intercept))
plt.plot(dist, intercept + slope*dist, 'r', label='fitted line')
plt.show()
X_nlos = X


mypath = './los_2_data_5_23._20/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
X = []
Y = []
vars = []

for file in onlyfiles:
    mat = sio.loadmat(mypath + file)
    rss_data = mat['file'][0][0][0][0][0][0]
    los_watt = rss_data[:,1]
    rss_data = 10 * np.log10(rss_data[:, 1])
    idx = ~np.isnan(rss_data)
    rss_data = rss_data[idx]
    los_watt = los_watt[idx]
    vars.append(np.var(rss_data))
    # print(np.var(rss_data, axis=0))
    X.extend(rss_data)  # RSS values
    Y.extend(np.tile(mat['file'][0, 0][0][-1][0][-1][0], reps=(len(rss_data), 1)))  # tile stationary location labels
print('Estimated LOS shadow fading variance: ',np.mean(vars))

X = np.array(X)
Y = np.array(Y)
rec_loc = np.array([0, 12.99])
dist = 10*np.log10(np.linalg.norm(rec_loc - Y, axis=1, ord=2))
slope, intercept, r_value, p_value, std_err = stats.linregress(dist,X)
print('Estimated LOS alpha: ',slope)
print('Estimated P0: ', intercept)
plt.scatter(dist,X)
plt.xlabel('10log10 Distance (m)')
plt.ylabel('Power (dBm)')
plt.rc('text', usetex=True) #  %f' % float(mse)
plt.title(r'LOS $\alpha=$ %f $\sigma=$ %f $P0=$ %f' % (slope, np.mean(vars), intercept))
plt.plot(dist, intercept + slope*dist, 'r', label='fitted line')
plt.show()
X_los = X



mypath = './grid_5_23_20/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
X = []
Y = []
vars = []

for file in onlyfiles:
    mat = sio.loadmat(mypath + file)
    rss_data = mat['file'][0][0][0][0][0][0]
    grid_watt = rss_data[:,1]
    rss_data = 10 * np.log10(rss_data[:, 1])
    idx = ~np.isnan(rss_data)
    rss_data = rss_data[idx]
    grid_watt = grid_watt[idx]
    vars.append(np.var(rss_data))
    # print(np.var(rss_data, axis=0))
    X.extend(rss_data)  # RSS values
    Y.extend(np.tile(mat['file'][0, 0][0][-1][0][-1][0], reps=(len(rss_data), 1)))  # tile stationary location labels
print('Estimated Grid shadow fading variance: ',np.mean(vars))

plt.hist(nlos_watt,color='b', density=True, alpha=0.5, label='nlos')
plt.hist(los_watt,color='g', density=True, alpha=0.5, label='los')
plt.hist(grid_watt,color='r', density=True, alpha=0.5, label='grid')
plt.xlabel('Watts')
plt.ylabel('density')
plt.legend()
plt.show()


plt.hist(X_nlos,color='b', density=True, alpha=0.5, label='nlos')
plt.hist(X_los,color='g', density=True, alpha=0.5, label='los')
plt.hist(X,color='r', density=True, alpha=0.5, label='grid')
plt.xlabel('dBm')
plt.ylabel('density')
plt.legend()
plt.show()