import itertools

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from sklearn import preprocessing
from tensorflow import keras
from scipy import spatial


from sklearn import mixture
import sklearn
print(sklearn.__version__)

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])

n_components = 20


def plot_results(X, Y_, means, covariances, index, weights):
    splot = plt.subplot(1, 1, 1)
    mse = []
    correct = 0
    incorrect = 0
    ms = []
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], s=15, color=color)
        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)
        plt.scatter(mean[0], mean[1], c='red', marker='s', s=100)
        # print(mean)
        ms.append(mean)

    tree = spatial.KDTree(np.array(ms))
    mse = np.mean(np.power(tree.query(np.unique(X, axis=0))[0], 2))


    plt.grid(True)
    # plt.xlim([0, 25])
    # plt.ylim([-2, 10])
    plt.xlabel('X-position (m)')
    plt.ylabel('Y-position (m)')
    return mse, correct


# import test set
Y_pred = np.load('CNN_preds_normal.npy')
y_t = np.load('CNN_true_normal.npy')  # (1000,2)


# 4 vehicles, every group of 16 (4*4) is 4 tires for 4 cars
print(y_t.shape)
# y_t = np.reshape(y_t, newshape=(8,4*2*8,2))
# Y_pred = np.reshape(Y_pred, newshape=(8,4*2*8,2))

# metrics
mse_over_time = []
num_clusters = []
cluster_correct = []
# for t in range(len(Y_pred)):
# Fit a Dirichlet process Gaussian mixture using five components
dpgmm = mixture.BayesianGaussianMixture(n_components=8, covariance_type='full', tol=1e-3, reg_covar=1e-06,
                                        max_iter=100, n_init=1, init_params='kmeans',
                                        weight_concentration_prior_type='dirichlet_process',
                                        weight_concentration_prior=None).fit(Y_pred)

##### PLOTTING ######
mse, correct = plot_results(y_t, dpgmm.predict(Y_pred), dpgmm.means_, dpgmm.covariances_, 1, dpgmm.weights_)

cluster_correct.append(correct/8.)
# weight_concentration_prior


plt.scatter(Y_pred[:,0], Y_pred[:,1], c='r', marker='.')
receviers = np.array([[7.5,0],[0,12.99],[15,12.99]])
# plt.scatter(receviers[:,0], receviers[:,1], marker='^', s=200)
plt.scatter(y_t[:,0], y_t[:,1], c='k',marker='s',s =100)
legend_elements = [Line2D([0], [0], marker='s', color='w', label='estimated',
                          markerfacecolor='red', markersize=15),
                   Line2D([0], [0], marker='s', color='w', label='true',
                          markerfacecolor='k', markersize=15)#,
                   # Line2D([0], [0], marker='^', color='w', label='receiver',
                   #        markerfacecolor='b', markersize=15)
                   ]
plt.legend(handles=legend_elements, loc='lower left')
# plt.title('DPGMM, Estimated number of clusters: %d' % len(np.unique(dpgmm.predict(Y_pred))))
plt.title('Mean Square Error of Cluster Means: %f' % float(mse))
plt.show()
num_clusters.append(len(np.unique(dpgmm.predict(Y_pred))))
for k, w in enumerate(dpgmm.weights_):
    plt.bar(k, w, width=0.9, color='#56B4E9', zorder=3,
            align='center', edgecolor='black')
    plt.text(k, w + 0.007, "%.1f%%" % (w * 100.),
             horizontalalignment='center')
plt.title('Weights')
plt.show()

# compute centroid MSE
print('MSE: ', mse)
# compute mixture accuracy
print('Average Number of Predicted Clusters: ', np.mean(num_clusters))
# compute classification accuracy
print('Cluster Classification accuracy: ',np.mean(cluster_correct))


