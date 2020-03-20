import itertools

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from sklearn import mixture

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])


def plot_results(X, Y_, means, covariances, index):
    splot = plt.subplot(1, 1, 1)
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
        plt.scatter(mean[0], mean[1], c='red', marker='*', s=30)

    plt.xticks(())
    plt.yticks(())
    plt.grid(True)
    plt.xlabel('X-position')
    plt.ylabel('Y-position')



# Number of samples per component
n_samples = 500

# Generate random sample, two components
centers = [[1, 1], [-1, -1], [1, -1], [5,7], [5,5], [7,5], [0,6],[0,8],[2,6]]
X, labels_true = make_blobs(n_samples=4*9, centers=centers, cluster_std=5.,
                            random_state=0)

# X = StandardScaler().fit_transform(X)


# Fit a Dirichlet process Gaussian mixture using five components
dpgmm = mixture.BayesianGaussianMixture(n_components=4*9,max_iter=1,
                                        covariance_type='full').fit(X)
plot_results(X, dpgmm.predict(X), dpgmm.means_, dpgmm.covariances_, 1)
# weight_concentration_prior
centers = np.array(centers)
plt.scatter(centers[:,0], centers[:,1], c='k',marker='s', s=30)

legend_elements = [Line2D([0], [0], marker='*', color='w', label='estimated',
                          markerfacecolor='red', markersize=15),
                   Line2D([0], [0], marker='s', color='w', label='true',
                          markerfacecolor='k', markersize=15)
                   ]
plt.legend(handles=legend_elements, loc='upper left')
plt.title('DPGMM, Estimated number of clusters: %d' % len(np.unique(dpgmm.predict(X))))
plt.show()




