import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial
from sklearn import mixture
import sklearn
from sklearn.metrics import f1_score
print(sklearn.__version__)


def plot_results(X, Y_, means, covariances, index, weights, Y_pred):
    correct = 0
    ms = []
    true_beacons = []
    indicators_correct = 0

    clus_count = 0
    for i, (mean, covar) in enumerate(zip(
            means, covariances)):

        if np.any(Y_ == i):
            correct_loc = np.unique(X, axis=0)[np.argmin(np.linalg.norm(np.unique(X, axis=0) - mean, ord=2, axis=1))]
            indicators_correct += np.sum(X[Y_ == i] == correct_loc)/3  # Number of correct indicators for this cluster. counts each axis so must div by 3

            true_beacons.append(correct_loc)
            ms.append(mean)
            clus_count += len(X[Y_ == i])  # number of indicators for this cluster

    tree = spatial.KDTree(np.array(ms))
    mse = np.mean(np.power(tree.query(np.unique(X, axis=0))[0], 2))

    return mse, correct, indicators_correct / clus_count

# Define bound
Pt = 44.8  # Watts
Pn = 1e-14  # watts
Gt = 25  # linear gain
Gr = 2.5  # linear gain
B = 2*10**6  # effective bandwidth
f = 1575*10**6  # carrier frequnecy
c = 3*10**8  # speed of light, m/s
W = c / f  # wavelength
rh = 10*10**3  # radius
lambda_a = 10**-12  # anchor density per volume with radius rh
# beacons constants
lambda_bs = np.linspace(10**-13, 10**-12)  # beacon density per volume with radius rh
# generate location estimates using bound
stationary_duration = 0.1  # how long the SOP lasts, seconds
lambda_s = 100  # how many signals are transmitted per stationary duration

# begin MC
averaging_iters = 30
num_clus = []
i_vals = []
mse_vs_n = []
indicators = []
count = 1
for lambda_b in lambda_bs:
    print('step ',count,' of ',len(lambda_bs))
    i_vals.append(lambda_b)
    for _ in range(averaging_iters):
        # redefine bound
        crlb = (8 * np.pi ** 2 * c ** 2 * rh ** 2 * Pn) / (B ** 2 * W ** 2 * Pt * Gt * Gr * (4 / 3 * np.pi * rh ** 3 * lambda_a - 1))
        # print(crlb)
        # generate beacons
        n_b = np.random.poisson(4 / 3 * np.pi * rh ** 3 * lambda_b)  # number of beacons
        # print('n_b: ',n_b)
        if n_b < 2:
            n_b = 2
        beacons = []  # Uniformly distribute n_b beacons over the rh sphere
        for _ in range(n_b):
            r = np.random.uniform(0, rh)  # 0 < r < rh
            theta = np.random.uniform(0, np.pi)  # 0 < theta < pi
            phi = np.random.uniform(0, 2 * np.pi)  # 0 < phi < 2pi
            beacons.append([r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)])
        beacons = np.array(beacons)

        # generate cluster data
        cluster_data = []
        truth = []
        for beacon in beacons:
            n_i = np.random.poisson(stationary_duration * lambda_s)  # how many location estimates for this beacon
            # print('n_i: ', n_i)
            if n_i == 0:
                n_i = 1
            data_to_add = np.random.normal(loc=beacon, scale=crlb, size=(n_i, 3))
            cluster_data.extend(data_to_add)
            truth.extend(np.repeat([beacon], n_i, axis=0))
        cluster_data = np.array(cluster_data)
        truth = np.array(truth)

        # metrics
        mse_over_time = []
        num_clusters = []
        cluster_correct = []
        # Fit a Dirichlet process Gaussian mixture using five components
        dpgmm = mixture.BayesianGaussianMixture(n_components=len(cluster_data), covariance_type='full', tol=1e-3, reg_covar=1e-06,
                                                max_iter=100, n_init=10, init_params='kmeans',
                                                weight_concentration_prior_type='dirichlet_process',
                                                weight_concentration_prior=None).fit(cluster_data)

        ##### PLOTTING ######
        mse, correct, indicators_correct = plot_results(truth, dpgmm.predict(cluster_data), dpgmm.means_, dpgmm.covariances_, 1, dpgmm.weights_, cluster_data)

        num_clusters.append(len(np.unique(dpgmm.predict(cluster_data))))
        num_clus.append(np.mean(num_clusters) / n_b)

        mse_vs_n.append(mse)

        indicators.append(indicators_correct)
    count += 1
    # Append predicted number of clusters, n_b
    # Append predicted indicators , true indicators
    # f1_score(s2_true_num_arr, s2_num_arr, average='weighted')?????????????????

# plotting
i_vals = np.array(i_vals) * (4/3 * np.pi * rh**3)
################ NUMBER OF ########################
plt.plot(i_vals,np.mean(np.reshape(num_clus, newshape=(int(len(num_clus)/averaging_iters), averaging_iters)), axis=-1))
plt.xlabel(r'Expected Number of Beacons $\frac{4}{3} \pi r_h^3 \lambda_b$')
plt.ylabel('Estimated Number of Beacons Accuracy (%)')
plt.show()
############### MSE ################################
plt.plot(i_vals,np.mean(np.reshape(mse_vs_n, newshape=(int(len(mse_vs_n)/averaging_iters), averaging_iters)), axis=-1))
plt.xlabel(r'Expected Number of Beacons $\frac{4}{3} \pi r_h^3 \lambda_b$')
plt.ylabel('Beacon Localization MSE (m)')
plt.show()
################ INDICATORS #############################
plt.plot(i_vals,np.mean(np.reshape(indicators, newshape=(int(len(indicators)/averaging_iters), averaging_iters)), axis=-1))
plt.xlabel(r'Expected Number of Beacons $\frac{4}{3} \pi r_h^3 \lambda_b$')
plt.ylabel('Indicator Variable Accuracy (%)')
plt.show()
