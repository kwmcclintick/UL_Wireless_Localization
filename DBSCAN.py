import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from matplotlib.lines import Line2D


if __name__ == '__main__':
    # #############################################################################
    # Generate sample data
    centers = [[1, 1], [-1, -1], [1, -1], [5,7], [5,5], [7,5], [0,6],[0,8],[2,6]]
    X, labels_true = make_blobs(n_samples=4*9, centers=centers, cluster_std=5.,
                                random_state=0)

    # X = StandardScaler().fit_transform(X)

    # #############################################################################
    # Compute DBSCAN
    db = DBSCAN(eps=3.5, min_samples=1).fit(X)

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(labels_true, labels))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(labels_true, labels))
    # print("Silhouette Coefficient: %0.3f"
    #       % metrics.silhouette_score(X, labels))

    # #############################################################################
    # Plot result
    import matplotlib.pyplot as plt

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markersize=5)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markersize=5)
    means = []
    for group in unique_labels:
        data = X[labels == group]
        means.append([np.mean(data[:, 0]), np.mean(data[:, 1])])
    means = np.array(means)
    centers = np.array(centers)
    plt.scatter(means[:, 0], means[:, 1], c='red', marker='*', s=45)
    plt.scatter(centers[:, 0], centers[:, 1], c='k', marker='s', s=45)
    legend_elements = [Line2D([0], [0], marker='*', color='w', label='estimated',
                              markerfacecolor='red', markersize=15),
                       Line2D([0], [0], marker='s', color='w', label='true',
                              markerfacecolor='k', markersize=15)
                       ]
    plt.legend(handles=legend_elements, loc='upper left')
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.xlabel('X-position')
    plt.ylabel('Y-position')
    plt.show()






