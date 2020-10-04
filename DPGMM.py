import itertools
import networkx as nx
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial
from chinese_whispers import chinese_whispers, aggregate_clusters
from itertools import combinations
from sklearn import mixture
import sklearn
print(sklearn.__version__)
from sklearn.metrics import f1_score

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])

n_components = 20


def plot_results(X, Y_, means, covariances, index, weights, Y_pred):
    # splot = plt.subplot(1, 1, 1)
    mse = []
    correct = 0
    incorrect = 0
    ms = []
    true_beacons = []
    f1_scores = []
    indicators_correct = 0
    indicates = 0

    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        # v, w = linalg.eigh(covar)
        # v = 2. * np.sqrt(2.) * np.sqrt(v)
        # u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        # if not np.any(Y_ == i):
        #     continue
        # plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], s=15, color=color)
        # Plot an ellipse to show the Gaussian component
        # angle = np.arctan(u[1] / u[0])
        # angle = 180. * angle / np.pi  # convert to degrees
        # ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        # ell.set_clip_box(splot.bbox)
        # ell.set_alpha(0.5)
        # splot.add_artist(ell)
        # plt.scatter(mean[0], mean[1], c='red', marker='s', s=100)
        # print(mean)
        if np.any(Y_ == i):
            # if a prediction is part of a cluster whos mean is closest to the predictions ground truth, its indicator variable is correct
            correct_loc = np.unique(X, axis=0)[np.argmin(np.linalg.norm(np.unique(X, axis=0) - mean, ord=2, axis=1))]
            indicators_correct += np.sum(X[Y_ == i] == correct_loc)/2
            true_beacons.append(correct_loc)
            ms.append(mean)
            indicates += len(X[Y_ == i])

            # OR F1 score
            # correct location gets set to 1
            f1_correct = np.ones(shape=(len(X[Y_ == i])))
            # other locations get sets to 0
            f1_preds = np.where(np.array(X[Y_ == i] == correct_loc)[:,0], 1, 0)

            f1_scores.extend(f1_score(f1_correct.astype(np.int), f1_preds.astype(np.int), average=None))



    np.save('DPGMM_means', ms)
    np.save('DPGMM_true_means',true_beacons)

    tree = spatial.KDTree(np.array(ms))
    mse = np.mean(np.power(tree.query(np.unique(X, axis=0))[0], 2))
    bias = np.mean(tree.query(np.unique(X, axis=0))[0])

    # plt.grid(True)
    # plt.xlim([0, 25])
    # plt.ylim([-2, 10])
    # plt.xlabel('X-position (m)')
    # plt.ylabel('Y-position (m)')
    return mse, bias, correct, np.mean(f1_scores)


truth = np.array([[5.9715, 6.7265], [5.3495, 9.2845]])  # true vehcile centers

averaging_iters = 2

bias_p = []
var_p = []
bias_mu = []
num_clus = []
cw_num_clus = []
i_vals = []
mse_vs_n = []
cw_mse_vs_n = []
Y_pred = np.load('bound_ests.npy')[:229]
indicators = []
cw_indicators = []
print('MC...')
for m_i in range(8, len(Y_pred)):
    i_vals.append(m_i)
    num_clusters = []
    for _ in range(averaging_iters):
        Y_pred = np.load('bound_ests.npy')[:229]
        idx = np.random.choice(len(Y_pred), size=m_i, replace=False)
        Y_pred = Y_pred[idx]
        y_t = np.load('CNN_true_normal.npy')  # (1000,2)
        y_t = y_t[idx]

        # bias, variance of location estimates
        bias_p.append(np.mean(Y_pred - y_t))
        var_p.append(np.mean(np.power(Y_pred - y_t, 2)))

        # metrics
        mse_over_time = []
        cluster_correct = []
        # Fit a Dirichlet process Gaussian mixture using five components
        dpgmm = mixture.BayesianGaussianMixture(n_components=m_i, covariance_type='full', tol=1e-3, reg_covar=1e-06,
                                                max_iter=1000, n_init=1, init_params='kmeans',
                                                weight_concentration_prior_type='dirichlet_process',
                                                weight_concentration_prior=None).fit(Y_pred)

        ##### PLOTTING ######
        mse, bias, correct, indicators_correct = plot_results(y_t, dpgmm.predict(Y_pred), dpgmm.means_, dpgmm.covariances_, 1, dpgmm.weights_, Y_pred)
        indicators.append(indicators_correct)
        cluster_correct.append(correct/8.)
        bias_mu.append(bias)
        # weight_concentration_prior


        # plt.scatter(Y_pred[:,0], Y_pred[:,1], c='r', marker='.')
        receviers = np.array([[7.5,0],[0,12.99],[15,12.99]])
        # plt.scatter(receviers[:,0], receviers[:,1], marker='^', s=200)
        # plt.scatter(y_t[:,0], y_t[:,1], c='k',marker='s',s =100)
        # legend_elements = [Line2D([0], [0], marker='s', color='w', label='estimated',
        #                           markerfacecolor='red', markersize=15),
        #                    Line2D([0], [0], marker='s', color='w', label='true',
        #                           markerfacecolor='k', markersize=15)#,
        #                    # Line2D([0], [0], marker='^', color='w', label='receiver',
        #                    #        markerfacecolor='b', markersize=15)
        #                    ]
        # plt.legend(handles=legend_elements, loc='lower left')
        # plt.title('DPGMM, Estimated number of clusters: %d' % len(np.unique(dpgmm.predict(Y_pred))))
        # plt.title('Mean Square Error of Cluster Means: %f' % float(mse))
        # plt.show()
        num_clusters.append(len(np.unique(dpgmm.predict(Y_pred))))
        # for k, w in enumerate(dpgmm.weights_):
        #     plt.bar(k, w, width=0.9, color='#56B4E9', zorder=3,
        #             align='center', edgecolor='black')
        #     plt.text(k, w + 0.007, "%.1f%%" % (w * 100.),
        #              horizontalalignment='center')
        # plt.title('Weights')
        # plt.show()

        # compute centroid MSE
        # print('MSE: ', mse)
        mse_vs_n.append(mse)
        # compute mixture accuracy
        # print('Average Number of Predicted Clusters: ', np.mean(num_clusters))
        num_clus.append(np.mean(num_clusters))

        # compute classification accuracy
        # print('Cluster Classification accuracy: ',np.mean(cluster_correct))

        # CW Copy paste #########################################################################################
        means = np.unique(np.load('DPGMM_means.npy'), axis=0)
        true_means = np.load('DPGMM_true_means.npy')

        G = nx.Graph()
        for i in range(len(means)):
            G.add_node(i, pos=means[i])

        weights = []
        for i in range(len(means)):
            for j in range(len(means)):
                if i != j:
                    G.add_edge(i, j, weight=-np.linalg.norm(means[i] - means[j], ord=2) ** len(means))
                    weights.append(G.edges()[i, j]['weight'])

        chinese_whispers(G, weighting='top')

        # Print the clusters in the descending order of size
        cluster_mean = []
        cw_ind_correct = 0
        for label, cluster in sorted(aggregate_clusters(G).items(), key=lambda e: len(e[1]), reverse=True):
            idxs = list(cluster)
            ms = []
            for idx in idxs:
                node_data = G.nodes().data()[idx]
                ms.append(node_data['pos'].tolist())

            ms = np.array(ms)  # beacon locations from this class
            # create an ordered lists of ground truths for ms
            truths = []
            for data in ms:
                truths.append(true_means[np.where(means==data, True, False)])
            truths = np.array(truths)

            test = np.mean(ms, axis=0) # cluster mean of this cluster
            cluster_mean.append(test)

            # if a node whose class's mean is closest to its ground truths mean, its in the correct class
            for data in truths:
                if np.array_equal(data, [4.223, 8.188]) or np.array_equal(data, [6.473, 10.392]) or np.array_equal([3.901, 9.906], data) or np.array_equal([6.801, 8.652], data):  # car with center at [5.3495, 9.2845]
                    if np.linalg.norm(truth[0] - test, ord=2) > np.linalg.norm(truth[1] - test, ord=2):
                        cw_ind_correct += 1
                if np.array_equal(data, [4.502, 7.275]) or np.array_equal(data, [4.9, 5.555]) or np.array_equal(data, [7.439, 6.175]) or np.array_equal(data, [7.045, 7.901]):  # car with center at [5.9715, 6.7265]
                    if np.linalg.norm(truth[0] - test, ord=2) < np.linalg.norm(truth[1] - test, ord=2):
                        cw_ind_correct += 1

        cw_indicators.append(cw_ind_correct / len(means))
        cluster_mean = np.array(cluster_mean)  # class mean locations

        if len(cluster_mean) > 2: # gives inf center_mse
            # combinations('ABCD', 2) --> AB AC AD BC BD CD
            center_mse = np.infty
            for combo in combinations(cluster_mean, 2):
                current_mse = np.mean((combo - truth) ** 2)
                if current_mse < center_mse:
                    center_mse = current_mse
        else:
            center_mse = np.mean((cluster_mean - truth) ** 2)

        cw_num_clus.append(len(cluster_mean))
        cw_mse_vs_n.append(center_mse)


plm_bias_p = []
plm_var_p = []
plm_bias_mu = []
plm_num_clus = []
plm_mse_vs_n = []
plm_i_vals = []
plm_cw_mse_vs_n = []
plm_cw_num_clus = []
plm_indicators = []
plm_cw_indicators = []
Y_pred = np.load('PLM_RLS_ests.npy')
print('PLM...')
for m_i in range(8, 229):
    plm_i_vals.append(m_i)
    for _ in range(averaging_iters):
        Y_pred = np.load('PLM_RLS_ests.npy')

        # import test set
        idx = np.random.choice(229, size=m_i, replace=False)
        Y_pred = Y_pred[idx]
        y_t = np.load('CNN_true_normal.npy')  # (1000,2)
        y_t = y_t[idx]

        # bias, variance of location estimates
        plm_bias_p.append(np.mean(Y_pred - y_t))
        plm_var_p.append(np.mean(np.power(Y_pred - y_t, 2)))

        # metrics
        mse_over_time = []
        num_clusters = []
        cluster_correct = []
        # Fit a Dirichlet process Gaussian mixture using five components
        dpgmm = mixture.BayesianGaussianMixture(n_components=m_i, covariance_type='full', tol=1e-3, reg_covar=1e-06,
                                                max_iter=1000, n_init=1, init_params='kmeans',
                                                weight_concentration_prior_type='dirichlet_process',
                                                weight_concentration_prior=None).fit(Y_pred)

        ##### PLOTTING ######
        mse, bias, correct, indicators_correct = plot_results(y_t, dpgmm.predict(Y_pred), dpgmm.means_, dpgmm.covariances_, 1, dpgmm.weights_, Y_pred)
        plm_indicators.append(indicators_correct)
        plm_bias_mu.append(bias)
        cluster_correct.append(correct/8.)
        plm_mse_vs_n.append(mse)

        receviers = np.array([[7.5,0],[0,12.99],[15,12.99]])
        num_clusters.append(len(np.unique(dpgmm.predict(Y_pred))))
        plm_num_clus.append(np.mean(num_clusters))

        # CW Copy paste #########################################################################################
        means = np.unique(np.load('DPGMM_means.npy'), axis=0)
        true_means = np.load('DPGMM_true_means.npy')

        G = nx.Graph()
        for i in range(len(means)):
            G.add_node(i, pos=means[i])

        weights = []
        for i in range(len(means)):
            for j in range(len(means)):
                if i != j:
                    G.add_edge(i, j, weight=-np.linalg.norm(means[i] - means[j], ord=2) ** len(means))
                    weights.append(G.edges()[i, j]['weight'])

        chinese_whispers(G, weighting='top')

        # Print the clusters in the descending order of size
        cluster_mean = []
        cw_ind_correct = 0
        for label, cluster in sorted(aggregate_clusters(G).items(), key=lambda e: len(e[1]), reverse=True):
            idxs = list(cluster)
            ms = []
            for idx in idxs:
                node_data = G.nodes().data()[idx]
                ms.append(node_data['pos'].tolist())

            ms = np.array(ms)  # beacon locations from this class
            # create an ordered lists of ground truths for ms
            truths = []
            for data in ms:
                truths.append(true_means[np.where(means == data, True, False)])
            truths = np.array(truths)

            test = np.mean(ms, axis=0)  # cluster mean of this cluster
            cluster_mean.append(test)

            # if a node whose class's mean is closest to its ground truths mean, its in the correct class
            for data in truths:
                if np.array_equal(data, [4.223, 8.188]) or np.array_equal(data, [6.473, 10.392]) or np.array_equal(
                        [3.901, 9.906], data) or np.array_equal([6.801, 8.652],
                                                                data):  # car with center at [5.3495, 9.2845]
                    if np.linalg.norm(truth[0] - test, ord=2) > np.linalg.norm(truth[1] - test, ord=2):
                        cw_ind_correct += 1
                if np.array_equal(data, [4.502, 7.275]) or np.array_equal(data, [4.9, 5.555]) or np.array_equal(data,
                                                                                                                [7.439,
                                                                                                                 6.175]) or np.array_equal(
                        data, [7.045, 7.901]):  # car with center at [5.9715, 6.7265]
                    if np.linalg.norm(truth[0] - test, ord=2) < np.linalg.norm(truth[1] - test, ord=2):
                        cw_ind_correct += 1

        plm_cw_indicators.append(cw_ind_correct / len(means))
        cluster_mean = np.array(cluster_mean)

        if len(cluster_mean) > 2: # gives inf center_mse
            # combinations('ABCD', 2) --> AB AC AD BC BD CD
            center_mse = np.infty
            for combo in combinations(cluster_mean, 2):
                current_mse = np.mean((combo - truth) ** 2)
                if current_mse < center_mse:
                    center_mse = current_mse
        else:
            center_mse = np.mean((cluster_mean - truth) ** 2)

        plm_cw_num_clus.append(len(cluster_mean))
        plm_cw_mse_vs_n.append(center_mse)

cnn_bias_mu = []
cnn_bias_p = []
cnn_var_p = []
cnn_num_clus = []
cnn_i_vals = []
cnn_cw_num_clus = []
cnn_mse_vs_n = []
cnn_cw_mse_vs_n = []
cnn_indicators = []
cnn_cw_indicators = []
Y_pred = np.load('CNN_preds_normal.npy')
print('CNN...')
for m_i in range(8, len(Y_pred)):
    cnn_i_vals.append(m_i)
    for _ in range(averaging_iters):
        Y_pred = np.load('CNN_preds_normal.npy')
        idx = np.random.choice(len(Y_pred), size=m_i, replace=False)
        Y_pred = Y_pred[idx]
        y_t = np.load('CNN_true_normal.npy')  # (1000,2)
        y_t = y_t[idx]

        # bias, variance of location estimates
        cnn_bias_p.append(np.mean(Y_pred - y_t))
        cnn_var_p.append(np.mean(np.power(Y_pred - y_t, 2)))

        mse_over_time = []
        num_clusters = []
        cluster_correct = []
        dpgmm = mixture.BayesianGaussianMixture(n_components=m_i, covariance_type='full', tol=1e-3, reg_covar=1e-06,
                                                max_iter=1000, n_init=1, init_params='kmeans',
                                                weight_concentration_prior_type='dirichlet_process',
                                                weight_concentration_prior=None).fit(Y_pred)

        mse, bias, correct, indicators_correct = plot_results(y_t, dpgmm.predict(Y_pred), dpgmm.means_, dpgmm.covariances_, 1, dpgmm.weights_, Y_pred)
        cnn_indicators.append(indicators_correct)
        cnn_bias_mu.append(bias)
        cnn_mse_vs_n.append(mse)
        cluster_correct.append(correct/8.)
        receviers = np.array([[7.5,0],[0,12.99],[15,12.99]])
        num_clusters.append(len(np.unique(dpgmm.predict(Y_pred))))
        cnn_num_clus.append(np.mean(num_clusters))

        # CW Copy paste #########################################################################################
        means = np.unique(np.load('DPGMM_means.npy'), axis=0)
        true_means = np.load('DPGMM_true_means.npy')

        G = nx.Graph()
        for i in range(len(means)):
            G.add_node(i, pos=means[i])

        weights = []
        for i in range(len(means)):
            for j in range(len(means)):
                if i != j:
                    G.add_edge(i, j, weight=-np.linalg.norm(means[i] - means[j], ord=2) ** len(means))
                    weights.append(G.edges()[i, j]['weight'])

        chinese_whispers(G, weighting='top')

        # Print the clusters in the descending order of size
        cluster_mean = []
        cw_ind_correct = 0
        for label, cluster in sorted(aggregate_clusters(G).items(), key=lambda e: len(e[1]), reverse=True):
            idxs = list(cluster)
            ms = []
            for idx in idxs:
                node_data = G.nodes().data()[idx]
                ms.append(node_data['pos'].tolist())

            ms = np.array(ms)  # beacon locations from this class
            # create an ordered lists of ground truths for ms
            truths = []
            for data in ms:
                truths.append(true_means[np.where(means == data, True, False)])
            truths = np.array(truths)

            test = np.mean(ms, axis=0)  # cluster mean of this cluster
            cluster_mean.append(test)

            # if a node whose class's mean is closest to its ground truths mean, its in the correct class
            for data in truths:
                if np.array_equal(data, [4.223, 8.188]) or np.array_equal(data, [6.473, 10.392]) or np.array_equal(
                        [3.901, 9.906], data) or np.array_equal([6.801, 8.652],
                                                                data):  # car with center at [5.3495, 9.2845]
                    if np.linalg.norm(truth[0] - test, ord=2) > np.linalg.norm(truth[1] - test, ord=2):
                        cw_ind_correct += 1
                if np.array_equal(data, [4.502, 7.275]) or np.array_equal(data, [4.9, 5.555]) or np.array_equal(data,
                                                                                                                [7.439,
                                                                                                                 6.175]) or np.array_equal(
                        data, [7.045, 7.901]):  # car with center at [5.9715, 6.7265]
                    if np.linalg.norm(truth[0] - test, ord=2) < np.linalg.norm(truth[1] - test, ord=2):
                        cw_ind_correct += 1

        cnn_cw_indicators.append(cw_ind_correct / len(means))
        cluster_mean = np.array(cluster_mean)

        if len(cluster_mean) > 2: # gives inf center_mse
            # combinations('ABCD', 2) --> AB AC AD BC BD CD
            center_mse = np.infty
            for combo in combinations(cluster_mean, 2):
                current_mse = np.mean((combo - truth) ** 2)
                if current_mse < center_mse:
                    center_mse = current_mse
        else:
            center_mse = np.mean((cluster_mean - truth) ** 2)

        cnn_cw_num_clus.append(len(cluster_mean))
        cnn_cw_mse_vs_n.append(center_mse)


# plotting
############### BIAS ################################
plt.plot(i_vals,np.mean(np.reshape(bias_p, newshape=(int(len(bias_p)/averaging_iters), averaging_iters)), axis=-1))
plt.plot(plm_i_vals,np.mean(np.reshape(plm_bias_p, newshape=(int(len(plm_bias_p)/averaging_iters), averaging_iters)), axis=-1))
plt.plot(cnn_i_vals,np.mean(np.reshape(cnn_bias_p, newshape=(int(len(cnn_bias_p)/averaging_iters), averaging_iters)), axis=-1))
plt.legend(['Simulated Path Loss','Path Loss','Finger Print'])
plt.grid(True)
plt.xlabel('Localized Packets (N)')
plt.ylabel(r'Estimated Packet Location $\hat{P}_i$ Bias (m)')
plt.show()

plt.plot(i_vals,np.mean(np.reshape(bias_mu, newshape=(int(len(bias_mu)/averaging_iters), averaging_iters)), axis=-1))
plt.plot(plm_i_vals,np.mean(np.reshape(plm_bias_mu, newshape=(int(len(plm_bias_mu)/averaging_iters), averaging_iters)), axis=-1))
plt.plot(cnn_i_vals,np.mean(np.reshape(cnn_bias_mu, newshape=(int(len(cnn_bias_mu)/averaging_iters), averaging_iters)), axis=-1))
plt.legend(['Simulated Path Loss','Path Loss','Finger Print'])
plt.grid(True)
plt.xlabel('Localized Packets (N)')
plt.ylabel(r'Estimated Beacon Location $\hat{\mu}_k$ Bias (m)')
plt.show()


# need vehicle clusters


############### Variance ################################
plt.plot(i_vals,np.mean(np.reshape(var_p, newshape=(int(len(var_p)/averaging_iters), averaging_iters)), axis=-1))
plt.plot(plm_i_vals,np.mean(np.reshape(plm_var_p, newshape=(int(len(plm_var_p)/averaging_iters), averaging_iters)), axis=-1))
plt.plot(cnn_i_vals,np.mean(np.reshape(cnn_var_p, newshape=(int(len(cnn_var_p)/averaging_iters), averaging_iters)), axis=-1))
plt.legend(['Simulated Path Loss','Path Loss','Finger Print'])
plt.grid(True)
plt.xlabel('Localized Packets (N)')
plt.ylabel(r'Estimated Packet Location $\hat{P}_i$ Variance (m)')
plt.show()


plt.plot(i_vals,np.mean(np.reshape(mse_vs_n, newshape=(int(len(mse_vs_n)/averaging_iters), averaging_iters)), axis=-1))
plt.plot(plm_i_vals,np.mean(np.reshape(plm_mse_vs_n, newshape=(int(len(plm_mse_vs_n)/averaging_iters), averaging_iters)), axis=-1))
plt.plot(cnn_i_vals,np.mean(np.reshape(cnn_mse_vs_n, newshape=(int(len(cnn_mse_vs_n)/averaging_iters), averaging_iters)), axis=-1))
plt.legend(['Simulated Path Loss','Path Loss','Finger Print'])
plt.grid(True)
plt.xlabel('Localized Packets (N)')
plt.ylabel(r'Estimated Beacon Location $\hat{\mu}_k$ Variance (m)')
plt.show()

plt.plot(i_vals,np.mean(np.reshape(cw_mse_vs_n, newshape=(int(len(cw_mse_vs_n)/averaging_iters), averaging_iters)), axis=-1))
plt.plot(plm_i_vals,np.mean(np.reshape(plm_cw_mse_vs_n, newshape=(int(len(plm_cw_mse_vs_n)/averaging_iters), averaging_iters)), axis=-1))
plt.plot(cnn_i_vals,np.mean(np.reshape(cnn_cw_mse_vs_n, newshape=(int(len(cnn_cw_mse_vs_n)/averaging_iters), averaging_iters)), axis=-1))
plt.legend(['Simulated Path Loss','Path Loss','Finger Print'])
plt.grid(True)
plt.xlabel('Localized Packets (N)')
plt.ylabel(r'Estimated Vehicle Location $\hat{\mu}_j$ Variance (m)')
plt.show()


################ K, J ########################
plt.plot(i_vals,np.mean(np.reshape(num_clus, newshape=(int(len(num_clus)/averaging_iters), averaging_iters)), axis=-1))
plt.plot(plm_i_vals,np.mean(np.reshape(plm_num_clus, newshape=(int(len(plm_num_clus)/averaging_iters), averaging_iters)), axis=-1))
plt.plot(cnn_i_vals,np.mean(np.reshape(cnn_num_clus, newshape=(int(len(cnn_num_clus)/averaging_iters), averaging_iters)), axis=-1))
plt.legend(['Simulated Path Loss','Path Loss','Finger Print'])
plt.grid(True)
plt.xlabel('Localized Packets (N)')
plt.ylabel(r'Estimated Beacons $\hat{K}$')
plt.show()

plt.plot(i_vals,np.mean(np.reshape(cw_num_clus, newshape=(int(len(cw_num_clus)/averaging_iters), averaging_iters)), axis=-1))
plt.plot(plm_i_vals,np.mean(np.reshape(plm_cw_num_clus, newshape=(int(len(plm_cw_num_clus)/averaging_iters), averaging_iters)), axis=-1))
plt.plot(cnn_i_vals,np.mean(np.reshape(cnn_cw_num_clus, newshape=(int(len(cnn_cw_num_clus)/averaging_iters), averaging_iters)), axis=-1))
plt.legend(['Simulated Path Loss','Path Loss','Finger Print'])
plt.grid(True)
plt.xlabel('Localized Packets (N)')
plt.ylabel(r'Estimated Vehicles $\hat{J}$')
plt.show()

################ F1 SCORES INDICATORS #############################
plt.plot(i_vals,np.mean(np.reshape(indicators, newshape=(int(len(indicators)/averaging_iters), averaging_iters)), axis=-1))
plt.plot(plm_i_vals,np.mean(np.reshape(plm_indicators, newshape=(int(len(plm_indicators)/averaging_iters), averaging_iters)), axis=-1))
plt.plot(cnn_i_vals,np.mean(np.reshape(cnn_indicators, newshape=(int(len(cnn_indicators)/averaging_iters), averaging_iters)), axis=-1))
plt.legend(['Simulated Path Loss','Path Loss','Finger Print'])
plt.grid(True)
plt.xlabel('Localized Packets (N)')
plt.ylabel(r'Packet Location Estimate Indicator Variable $\hat{c}_i$ F1 Score')
plt.show()

plt.plot(i_vals,np.mean(np.reshape(cw_indicators, newshape=(int(len(cw_indicators)/averaging_iters), averaging_iters)), axis=-1))
plt.plot(plm_i_vals,np.mean(np.reshape(plm_cw_indicators, newshape=(int(len(plm_cw_indicators)/averaging_iters), averaging_iters)), axis=-1))
plt.plot(cnn_i_vals,np.mean(np.reshape(cnn_cw_indicators, newshape=(int(len(cnn_cw_indicators)/averaging_iters), averaging_iters)), axis=-1))
plt.legend(['Simulated Path Loss','Path Loss','Finger Print'])
plt.grid(True)
plt.xlabel('Localized Packets (N)')
plt.ylabel(r'Beacon Indicator Variable $\hat{w}_i$ F1 Score')
plt.show()
