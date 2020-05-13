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


from sklearn import mixture
import sklearn
print(sklearn.__version__)

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])

n_components = 4


def plot_results(X, Y_, means, covariances, index, weights):
    splot = plt.subplot(1, 1, 1)
    mse = []
    correct = 0
    incorrect = 0
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
        plt.scatter(mean[0], mean[1], c='red', marker='*', s=300)
        # plot car centroids
        plt.scatter(np.mean(X[0:4,0]), np.mean(X[0:4,1]), c='k', marker='*', s=300)
        plt.scatter(np.mean(X[4:, 0]), np.mean(X[4:, 1]), c='k', marker='*', s=300)

        # determine which car is closest to this centroid
        d_to_c1 = np.linalg.norm(np.array(np.subtract([np.mean(X[0:4,0]), np.mean(X[0:4,1])], [mean[0], mean[1]])), ord=1)
        d_to_c2 = np.linalg.norm(np.array(np.subtract([np.mean(X[4:, 0]), np.mean(X[4:, 1])], [mean[0], mean[1]])), ord=1)
        # cluster points
        c_pts = np.concatenate((np.expand_dims(X[Y_ == i, 0],axis=1), np.expand_dims(X[Y_ == i, 1], axis=1)), axis=1)

        test = np.argwhere(Y_ == i).flatten()
        if d_to_c1 > d_to_c2:
            mse.append(d_to_c2)
            if 0 in test:
                incorrect += 1
            if 1 in test:
                incorrect += 1
            if 2 in test:
                incorrect += 1
            if 3 in test:
                incorrect += 1
            if 4 in test:
                correct += 1
            if 5 in test:
                correct += 1
            if 6 in test:
                correct += 1
            if 7 in test:
                correct += 1
        else:
            mse.append(d_to_c1)
            if 0 in test:
                correct += 1
            if 1 in test:
                correct += 1
            if 2 in test:
                correct += 1
            if 3 in test:
                correct += 1
            if 4 in test:
                incorrect += 1
            if 5 in test:
                incorrect += 1
            if 6 in test:
                incorrect += 1
            if 7 in test:
                incorrect += 1

    plt.grid(True)
    plt.xlim([0, 25])
    plt.ylim([-2, 10])
    plt.xlabel('X-position (m)')
    plt.ylabel('Y-position (m)')
    return mse, correct


# import test set
Y_pred = np.load('CNN_preds.npy')
y_t = np.load('CNN_true.npy')  # (1000,2)
# 4 vehicles, every group of 16 (4*4) is 4 tires for 4 cars
print(y_t.shape)
y_t = np.reshape(y_t, newshape=(60,8,2))
Y_pred = np.reshape(Y_pred, newshape=(60,8,2))

# metrics
mse_over_time = []
num_clusters = []
cluster_correct = []
for t in range(len(Y_pred)):
    # Fit a Dirichlet process Gaussian mixture using five components
    dpgmm = mixture.BayesianGaussianMixture(n_components=n_components, weight_concentration_prior=1000.).fit(Y_pred[t])

    ##### PLOTTING ######
    mse, correct = plot_results(y_t[t], dpgmm.predict(Y_pred[t]), dpgmm.means_, dpgmm.covariances_, 1, dpgmm.weights_)
    for item in mse:
        mse_over_time.append(item)
    cluster_correct.append(correct/8.)
    # weight_concentration_prior

    plt.scatter(y_t[t,:,0], y_t[t,:,1], c='k',marker='s')
    plt.scatter(Y_pred[t,:,0], Y_pred[t,:,1], c='r', marker='s')

    legend_elements = [Line2D([0], [0], marker='*', color='w', label='estimated',
                              markerfacecolor='red', markersize=15),
                       Line2D([0], [0], marker='s', color='w', label='true',
                              markerfacecolor='k', markersize=15)
                       ]
    plt.legend(handles=legend_elements, loc='upper left')
    plt.title('DPGMM, Estimated number of clusters: %d' % len(np.unique(dpgmm.predict(Y_pred[t]))))
    plt.show()
    num_clusters.append(len(np.unique(dpgmm.predict(Y_pred[t]))))

    for k, w in enumerate(dpgmm.weights_):
        plt.bar(k, w, width=0.9, color='#56B4E9', zorder=3,
                align='center', edgecolor='black')
        plt.text(k, w + 0.007, "%.1f%%" % (w * 100.),
                 horizontalalignment='center')
    plt.title('Weights')
    plt.show()

# compute centroid MSE
print('MSE: ', np.mean(np.power(mse_over_time, 2)))
# compute mixture accuracy
print('Average Number of Predicted Clusters: ', np.mean(num_clusters))
# compute classification accuracy
print('Cluster Classification accuracy: ',np.mean(cluster_correct))


