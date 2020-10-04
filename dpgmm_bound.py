"""

"""


from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as sp
import scipy
import numpy as np
from scipy import linalg
import matplotlib as mpl
from matplotlib.patches import Patch
from scipy import spatial
from sklearn import mixture
import itertools


color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                                      'darkorange'])

'''

'''
def plot_results(X, Y_, means, covariances, index, title, P):
    # splot = plt.subplot(1, 1, 1)
    indicators_correct = 0
    true_beacons = []
    ms = []
    class_i = []
    loc_i = []
    truths = []
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
        # plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        # ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        # splot.add_artist(ell)

        if np.any(Y_ == i):
            # if a prediction is part of a cluster whos mean is closest to the predictions ground truth, its indicator variable is correct
            correct_loc = np.unique(X, axis=0)[np.argmin(np.linalg.norm(np.unique(X, axis=0) - mean, ord=2, axis=1))]
            indicators_correct += np.sum(X[Y_ == i] == correct_loc)/2
            true_beacons.append(correct_loc)
            ms.append(mean)

            tree = spatial.KDTree(np.array(ms))
            mse = np.mean(np.power(tree.query(np.unique(X, axis=0))[0], 2))
            bias = np.mean(tree.query(np.unique(X, axis=0))[0])

            class_i.extend(i*np.ones(len(X[Y_ == i])))  # predictions of this value
            loc_i.extend(X[Y_ == i])  # are believed to be from this beacon
            truths.extend(X[Y_ == i])

    return bias, mse, indicators_correct, np.array(class_i), np.array(loc_i), np.array(truths)


'''
'''
def RLS(anchors, d):
    tol = 1e-2
    p = np.mean(anchors, axis=0)  # initial guess

    estimated_error = 0
    for j in range(len(anchors)):
        estimated_error += np.abs((anchors[j, 0] - p[0]) ** 2 + (anchors[j, 1] - p[1]) ** 2 - d[j] ** 2)

    while np.linalg.norm(estimated_error) > tol:
        Jac = np.zeros((len(anchors), 2))
        f = np.zeros((len(anchors), 1))

        for j in range(len(anchors)):
            Jac[j, :] = -2 * (anchors[j, :] - p)
            f[j] = (anchors[j, 0] - p[0]) ** 2 + (anchors[j, 1] - p[1]) ** 2 - d[j] ** 2

        estimated_error = np.matmul(np.matmul(-np.linalg.inv(np.matmul(Jac.T, Jac)), Jac.T), f).T[0]
        p = p + estimated_error

    return p


# initialize plot arrays (function of?...)
iter = 2

s1_bias_arr = []
s1_var_arr = []
s2_f1_arr = []
s2_acc_arr = []
s2_bias_arr = []
s2_var_arr = []
s2_cluster_f1_arr = []


# Anchor/beacon parameters
# num_anchors = np.linspace(3,10, 8, endpoint=True)
# anchor_radii = np.linspace(5,25, 21, endpoint=True)
d_a = 20  # radius of circle of anchors in meters
lambda_bs = np.linspace(.001,.1, 20) / d_a

# Signal generation/detection parameters
# lambda_sops = np.linspace(100,10000, 20)

# PLM parameters
# shadow_fading = np.linspace(0.5, 8, 20)
# P0s = np.linspace(-40, -120, 20)
# alphas_plm = np.linspace(2, 5, 20)

in_val = lambda_bs  # Variable being plotted against


for value in in_val:
    s2_num_arr = []
    s2_true_num_arr = []
    for _ in range(iter):
        # CREATE ANCHORS
        areaTotal = np.pi * d_a ** 2
        lambda0 = value  # intensity (ie mean density) of the Poisson process
        numbPoints = sp.poisson(lambda0 * areaTotal).rvs()  # Poisson number of points
        if numbPoints < 3:  # prevent there from being less than 3 anchors
            numbPoints = 3
        r = d_a * np.random.rand(numbPoints)
        theta = 2 * np.pi * np.random.rand(numbPoints)
        x, y = r * np.cos(theta), r * np.sin(theta)
        anchors = np.array([list(a) for a in zip(x, y)])

        # CREATE BEACONS
        # areaTotal = np.pi*d_a**2
        # numbPoints = sp.poisson(lambda0 * areaTotal).rvs()  # Poisson number of points
        # if numbPoints < 2:  # prevent there from being no beacons
        #     numbPoints = 2
        # r = d_a*np.random.rand(numbPoints)
        # theta = 2*np.pi*np.random.rand(numbPoints)
        # x, y = r * np.cos(theta), r * np.sin(theta)
        # beacons = np.array([list(a) for a in zip(x, y)])
        beacons = np.array([[0, 0]])
        K = len(beacons)  # number of clusters

        # CREATE EMISSIONS
        P0 = -70  # reference dBm of some system
        d0 = 1  # reference distance (m)
        alpha_plm = 4  # path loss gradient for path loss model
        sigma_F = 3.5  # shadow fading SD (dB)

        alpha_grlt = 0.95  # GRLT detectors false positive constraint
        n_grlt = 1e4  # how many samples does the GRLT have to form a hypothesis? related to signal duration and sampling rate
        P_noise = .001 * 10**(-84/10)  # at worst, noise floor for UWB (1GHz BW) is -84 dBm

        duration = 0.1  # physically stationary time period
        lambda_s = 1000  # intensity of how frequently signals are transmitted

        X = []  # estimated, noisy anchor-beacon range data set
        Y = []  # beacon location data set

        ds = []
        pds = []
        rssis = []
        print('generating signals...')
        for k in range(K):
            N_k = sp.poisson(lambda_s * duration).rvs()  # number of emissions from the kth beacon
            add_count = 0  # How many detections occured for this anchor-beacon pair
            for n in range(N_k):
                add_flag = True  # emissions detected by all anchors are added to the data set
                RSSI = []
                for a in anchors:
                    d = np.linalg.norm(a-beacons[k], ord=2)  # distance between the anchor beacon pair
                    rssi_db = P0 - 10*alpha_plm*np.log10(d/d0) + np.random.normal(0, sigma_F)  # RSSI in dB
                    rssis.append(rssi_db)
                    RSSI.append(rssi_db)
                    rssi_W = .001 * 10**(rssi_db/10)  # RSSI in Watts

                    nu_1 = 1  # degrees of freedom within
                    nu_2 = n_grlt - 1  # degrees of freedom without
                    v = sp.f.isf(1-alpha_grlt, nu_1, nu_2)  # NP decision threshold
                    lambda_grlt = n_grlt * rssi_W/P_noise  # non-centrality parameter
                    PD = 1 - scipy.special.ncfdtr(nu_1, nu_2, lambda_grlt, v)  # probability of detection for this signal between this anchor-beacon pair
                    pds.append(PD)
                    ds.append(d)

                    if np.random.uniform(0, 1) > PD:  # if any anchor fails to detect, don't localize
                        add_flag = False
                        if N_k - n < 3 and add_count < 2:  # failsafe, always have at least 2 localizations per beacon
                            add_flag = True  # if so, allows a miss to become a hit
                if add_flag:
                    add_count += 1
                    distances = d0*np.power(10, (P0-np.array(RSSI))/(10*alpha_plm))  # using a PLM estimate distance from RSSI
                    X.append(distances)
                    Y.append(beacons[k])

        X = np.array(X)
        Y = np.array(Y)

        # plt.scatter(ds, pds)
        # plt.xlabel('Transmission Distance (m)')
        # plt.ylabel(r'$P_D$')
        # plt.grid(True)
        # plt.show()

        # ESTIMATE LOCATIONS FROM RANGES -- MULTILATERATION
        P = []  # position estimate data set
        print('Computing RLS estimates...')
        for i in range(len(X)):
            P.append(RLS(anchors, X[i]))
        P = np.array(P)

        s1_bias = np.mean((P-Y))
        s1_var = np.mean((P-Y)**2)

        s1_bias_arr.append(s1_bias)
        s1_var_arr.append(s1_var)

        print('S1 Bias: ', s1_bias)
        print('S1 Variance', s1_var)

        # # PLOTTING
        # plt.scatter(anchors[:,0], anchors[:,1])
        # plt.scatter(P[:,0], P[:,1])
        # plt.scatter(beacons[:, 0], beacons[:, 1])
        # plt.xlabel('X Location (m)')
        # plt.ylabel('Y Location (m)')
        # plt.grid()
        # plt.legend(['Anchor','Estimates','Beacon'])
        # # plt.legend(['Beacon','Anchor'])
        # plt.show()

        # # DPGMM
        # dpgmm = mixture.BayesianGaussianMixture(n_components=len(P), covariance_type='full', tol=1e-3, reg_covar=1e-06,
        #                                                 max_iter=100, n_init=1, init_params='kmeans',
        #                                                 weight_concentration_prior_type='dirichlet_process',
        #                                                 weight_concentration_prior=None).fit(P)
        #
        # bias, mse, indicators_correct, class_i, loc_i, truths = plot_results(Y, dpgmm.predict(P), dpgmm.means_, dpgmm.covariances_, 1, 'title', P)
        # num_clusters = len(np.unique(dpgmm.predict(P)))
        #
        # # PLOTTING
        # # plt.scatter(anchors[:, 0], anchors[:, 1])
        # # plt.scatter(beacons[:, 0], beacons[:, 1])
        # # plt.scatter(P[:, 0], P[:, 1])
        # # plt.xlabel('X Location (m)')
        # # plt.ylabel('Y Location (m)')
        # # plt.grid()
        # # plt.show()
        #
        # from sklearn.metrics import f1_score
        #
        # # there is almost certainly a better way to map this
        # class_Y = np.zeros(len(Y))
        # for i in range(len(Y)):
        #     for j in range(len(loc_i)):
        #         if np.array_equal(truths[i],loc_i[j]):
        #             class_Y[i] = class_i[j]
        #
        # f1 = f1_score(class_Y, class_i, average='weighted')
        # print('S2 Class F1 score: ', f1)
        # # compute classification accuracy
        # acc = np.sum(class_i == class_Y)/len(class_i)
        # print('S2 Class ACC: ', acc)
        #
        # # compute centroid MSE
        # print('S2 Bias: ', bias)
        # print('S2 Variance: ', mse)
        # # compute mixture accuracy
        # print('S2 # clusters: ', num_clusters)
        # print('True clusters: ', len(np.unique(Y, axis=0)))
        #
        # s2_acc_arr.append(acc)
        # s2_bias_arr.append(bias)
        # s2_var_arr.append(mse)
        # s2_f1_arr.append(f1)
        # s2_num_arr.append(num_clusters)
        # s2_true_num_arr.append(len(np.unique(Y, axis=0)))

    # f1_clus = f1_score(s2_true_num_arr, s2_num_arr, average='weighted')
    # s2_cluster_f1_arr.append(f1_clus)
    # print('Estimaed number of clusters F1 score: ', f1_clus)



# # average over iterations
plot_s1_bias = np.mean(np.reshape(s1_bias_arr, newshape=(int(len(s1_bias_arr) / iter), iter)), axis=-1)
plot_s1_var = np.mean(np.reshape(s1_var_arr, newshape=(int(len(s1_var_arr) / iter), iter)), axis=-1)
#
# plot_s2_acc = np.mean(np.reshape(s2_acc_arr, newshape=(int(len(s2_acc_arr) / iter), iter)), axis=-1)
# plot_s2_bias = np.mean(np.reshape(s2_bias_arr, newshape=(int(len(s2_bias_arr) / iter), iter)), axis=-1)
# plot_s2_var = np.mean(np.reshape(s2_var_arr, newshape=(int(len(s2_var_arr) / iter), iter)), axis=-1)
# plot_s2_f1 = np.mean(np.reshape(s2_f1_arr, newshape=(int(len(s2_f1_arr) / iter), iter)), axis=-1)
# plot_s2_clus_f1 = s2_cluster_f1_arr





plt.plot(in_val, plot_s1_bias)
plt.plot(in_val, plot_s1_var)
# plt.plot(in_val, plot_s2_bias)
# plt.plot(in_val, plot_s2_var)
# plt.plot(in_val, plot_s2_f1)
# plt.plot(in_val, plot_s2_acc)
# plt.plot(in_val, plot_s2_clus_f1)
plt.xlabel(r'Anchor Intensity $\lambda$ $(m^{-2})$')
plt.grid(True)
# plt.legend([r'$\hat{P}$ Bias (m)',r'$\hat{P}$ Variance',r'$\hat{\mu}_k$ Bias (m)',r'$\hat{\mu}_k$ Variance',
#             r'$\hat{c}_i$ F1 Score',r'$\hat{K}$ F1 Score'])
plt.legend([r'$\hat{P}$ Bias (m)',r'$\hat{P}$ Variance'])
plt.show()
















