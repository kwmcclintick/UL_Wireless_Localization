import numpy as np
import math
import matplotlib.pyplot as plt

#
# distance = np.logspace(start=0, stop=2, num=1000)
# P0 = -59.78 #dBm
# txPwrVar = 9.391907167846293 / 8. #dB
# # txPwrVar = 0
# NLOS_grad = -3.850275788138077
# #Account for variation in Tx Power
# norm = np.random.normal(0, 1, size=(len(distance)))
# P0var = P0 + norm * txPwrVar
# RSS = P0var + 10 * NLOS_grad * np.log10(distance)
#
#
#
# est_distance = np.power(10, (RSS - P0)/(10. * NLOS_grad))  # estimated distance to each receiver
# plt.semilogx(distance, RSS)
# plt.show()
#
# plt.semilogy(est_distance)
# plt.semilogy(distance)
# plt.show()
#
# mse = np.mean((est_distance-distance)**2)
# print(mse)

mses = []
loc = []
dists = []
for _ in range(int(232/8)):

    receivers = np.array([[7.5, 0], [0, 12.99], [15, 12.99]])

    locs = np.unique(np.load('CNN_true_normal.npy'),axis=0 )


    true_dists = []
    for i in range(len(locs)):
        dist = np.linalg.norm(locs[i]-receivers, axis=1, ord=2)
        true_dists.append(dist)

    true_dists = np.array(true_dists)


    P0 = -59.78 #dBm
    txPwrVar = 9.391907167846293  #dB
    # txPwrVar = 0
    NLOS_grad = -3.850275788138077
    #Account for variation in Tx Power
    norm = np.random.normal(0, 1, size=(len(true_dists), len(true_dists[0])))
    P0var = P0 + norm * txPwrVar
    RSS = P0var + 10 * NLOS_grad * np.log10(true_dists)


    est_distance = np.power(10, (RSS - P0)/(10. * NLOS_grad))  # estimated distance to each receiver


    mse = np.mean((est_distance-true_dists)**2)
    mses.append(mse)

    dists.extend(est_distance)
    loc.extend(locs)

loc = np.array(loc)
dists = np.array(dists)
print(loc.shape)
print(dists.shape)
np.save('bound_true_locs', loc)
np.save('bound_est_dists', dists)

