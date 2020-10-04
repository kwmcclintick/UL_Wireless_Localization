"""
Compute the CRLB of TDOA measurements for both constant variance and location dependent variance models
"""

import numpy as np
import matplotlib.pyplot as plt


'''
Compute the gradient of measurement covariance with respect to beacon location
'''
def dN_dtheta(anchors, beacon, sigma_0, N):
    # X sigma_0?
    ref = anchors[0]
    anchors = anchors[1::]
    dN_dtheta = []
    temp_ref = (beacon - ref) / np.linalg.norm(beacon-ref, ord=2)
    temp = []
    for a in anchors:
        temp.append((beacon - a) / np.linalg.norm(beacon-a, ord=2))

    temp = np.array(temp)

    # print(temp_ref)
    # print(np.ones((N-1,N-1)).shape)
    # print(np.diag(temp[:,0]).shape)

    for dim in range(3):
        dN_dtheta.append( temp_ref[dim]*np.ones((N-1,N-1)) + np.diag(temp[:,dim]) )
    return sigma_0 * np.array(dN_dtheta)

'''
Compute the gradient of measurements with respect to beacon location
'''
def dm_dtheta(anchors, beacon):
    c = 3 * 10 ** 8  # speed of light
    ref = anchors[0]
    anchors = anchors[1::]
    dm_dtheta = []
    for a in anchors:
        dm_dtheta.append( ( (beacon - a) / np.linalg.norm(beacon-a, ord=2) -
                                   (beacon - ref) / np.linalg.norm(beacon-ref, ord=2) ) )
    return np.array(dm_dtheta)

pts = 3
x_vals = np.linspace(-15000, 35000, pts)
beacons = np.concatenate((np.expand_dims(x_vals,axis=-1),
                          np.zeros((pts, 1)), np.zeros((pts, 1))), axis=1)  # beacon location
# anchors = np.array([[-1800, -800, 10], [-1100, 7200, 40], [8600, -7000, -25], [22000, 300, -70]])  # anchor locations, [x,y,z]

# CREATE ANCHORS
import scipy.stats as sp
d_a = 20000  #
areaTotal = 4/3 * np.pi * d_a ** 3
lambda0 = 1 / areaTotal  # intensity (ie mean density) of the Poisson process
numbPoints = sp.poisson(lambda0 * areaTotal).rvs()  # Poisson number of points
print(numbPoints)
if numbPoints < 4:  # prevent there from being less than 4 anchors
    numbPoints = 4

r = d_a * np.random.rand(numbPoints)
theta = 2 * np.pi * np.random.rand(numbPoints)
phi = 2 * np.pi * np.random.rand(numbPoints)
x, y, z = r * np.cos(theta) * np.sin(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(phi)
anchors = np.array([list(a) for a in zip(x, y, z)])
N = len(anchors)

sigmas = np.array([.5])
plot_crlbs = []
for sig in sigmas:
    sigma_toa = sig  # m at 10km
    N_tdoa = sigma_toa * np.ones(shape=(N-1,N-1)) + sigma_toa * np.eye(N-1)  # N-1 X N-1

    crlbs = []
    for beacon in beacons:
        # compute CRLB (6) with only first term
        dm_dt = dm_dtheta(anchors, beacon)  # N-1 X 3
        theta = np.expand_dims(beacon, axis=-1)  # 3 X 1
        J_theta = np.matmul(np.matmul(dm_dt.T, np.linalg.inv(N_tdoa)), dm_dt)  # 3 X 3
        CRLB = np.linalg.inv(J_theta)  # 3 X 3
        crlbs.append(CRLB)  # diag of the CRLB gives x,y,z CRLB values

    plot_crlbs.append(crlbs)

dist_crlbs = []
for beacon in beacons:
    # compute CRLB (5) with both terms
    dm_dt = dm_dtheta(anchors, beacon)  # N-1 X 3
    theta = np.expand_dims(beacon, axis=-1)  # 1 X 3

    # how is sigma as a function of theta defined???
    sigma_0 = .5 / 5000  # This zero-intercept slope gives a 0.5 m std error at 10km as per the paper's specs
    sigma_1 = sigma_0 * np.linalg.norm(beacon - anchors[0], ord=2)
    N_tdoa = sigma_1 * np.ones(shape=(N - 1, N - 1)) + np.diag(sigma_0 * np.linalg.norm(beacon - anchors[1::], axis=1, ord=2))  # N-1 X N-1
    dN_dt = dN_dtheta(anchors, beacon, sigma_0, N)  # 3 X N-1 X N-1
    # print('step1')
    # print(np.linalg.pinv(N_tdoa).shape)
    # print(dN_dt.shape)
    # print(np.matmul(np.linalg.pinv(N_tdoa), dN_dt).shape)
    # print('step2')
    # print(np.matmul(np.linalg.pinv(N_tdoa), dN_dt).shape)
    # print(np.linalg.pinv(N_tdoa).shape)
    # print(np.matmul(np.matmul(np.linalg.pinv(N_tdoa), dN_dt), np.linalg.pinv(N_tdoa)).shape)
    # print('step3')
    # print(np.matmul(np.matmul(np.linalg.pinv(N_tdoa), dN_dt), np.linalg.pinv(N_tdoa)).shape)
    # print(dN_dt.shape)
    trace = np.matmul(np.matmul(np.matmul(np.linalg.pinv(N_tdoa), dN_dt), np.linalg.pinv(N_tdoa)), dN_dt)
    # print(trace.shape)

    traces = []
    for t in trace:
        traces.append(np.sum(np.diag(t)))
    traces = np.diag(np.array(traces))

    J_theta = np.matmul(np.matmul(dm_dt.T, np.linalg.pinv(N_tdoa)), dm_dt) + 0.5*traces  # 3 X 3
    CRLB = np.linalg.pinv(J_theta)  # 3 X 3
    dist_crlbs.append(CRLB)  # diag of the CRLB gives x,y,z CRLB values


plot_crlbs = np.array(plot_crlbs)
dist_crlbs = np.array(dist_crlbs)

# generate some location estimtes around each beacon location based on the CRLBS
num_estimates = 100
ests = []
for i in range(len(beacons)):
    ests.extend(np.random.multivariate_normal(beacons[i], dist_crlbs[i], size=num_estimates))
ests = np.array(ests)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(anchors[:,0], anchors[:,1], anchors[:,2])
ax.scatter(ests[:,0], ests[:,1], ests[:,2], alpha=.25, marker='x')
ax.scatter(beacons[:,0], beacons[:,1], beacons[:,2], marker='s')
ax.legend(['Anchors','Location Estimates','Beacons'])
plt.show()




# Plot CRLB for
for item in plot_crlbs:
    plt.plot(x_vals, item[:,0,0])
plt.plot(x_vals, dist_crlbs[:,0,0])
plt.xlabel('X-coord (m) $y=0$, $z=0$')
plt.ylabel('CRLB (m)')
plt.grid(True)
plt.ylim([0, 500])
plt.legend([r'$\sigma_{TOF}=0.5$ m', r'$\sigma_{TOF}(\theta)$'])
# plt.legend([r'$\sigma_{TOF}=0.3$ m',r'$\sigma_{TOF}=0.5$ m', r'$\sigma_{TOF}=0.8$ m'])
plt.show()









