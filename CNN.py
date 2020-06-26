"""


"""


import math
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization
from keras.models import Sequential
from keras import optimizers
import matplotlib.pyplot as plt
import scipy.stats as stats
import keras
from sklearn import preprocessing
import scipy.io as sio
from os import listdir
from sklearn.model_selection import StratifiedKFold
from os.path import isfile, join
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Conv2D, MaxPool2D, Dropout, Flatten, Dense, Concatenate, Activation, BatchNormalization, AveragePooling2D
from keras.models import Model


def res_block(input_layer, filters, depth=1):
    shortcut = input_layer
    output_layer = input_layer
    for i in range(depth):
        output_layer = Conv2D(filters=filters, kernel_size=(3,1), padding='same')(output_layer)
        output_layer = BatchNormalization(axis=3)(output_layer) # AXIS=3?????
        if i + 1 < depth:  # wait to do the last relu activation until after concatenation
            output_layer = Activation('relu')(output_layer)
    # output_layer = MaxPool2D(pool_size=(2, 1), padding="same")(output_layer)
    # output_layer = Dropout(0.0)(output_layer)
    output_layer = Concatenate(axis=-1)([output_layer, shortcut])  # CONCAT OR ADD???
    output_layer = Activation('relu')(output_layer)
    return output_layer




def define_model(nn, l, bn, do):
    IMAGE_WIDTH = 3
    IMAGE_HEIGHT = 1
    depth = 1
    inputs = Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 1))
    conv1 = Conv2D(filters=512, kernel_size=(3, 1), padding='same', activation='relu')(inputs)
    conv2 = res_block(conv1, 256, depth)
    conv3 = res_block(conv2, 128, depth)
    conv4 = res_block(conv3, 64, depth)
    conv5 = res_block(conv4, 32, depth)
    conv6 = res_block(conv5, 16, depth)
    conv7 = res_block(conv6, 8, depth)
    conv8 = res_block(conv7, 4, depth)
    conv9 = res_block(conv8, 2, depth)
    flatten = Flatten()(conv9)
    outputs = Dense(256, activation='relu')(flatten)
    outputs = Dense(2, activation='linear')(outputs)

    model = Model(inputs=inputs, outputs=outputs)



    # model = keras.Sequential()
    # filt_size = 3
    # p_size = 2
    # do = 0.0
    # p = 3
    #
    # # conv layers 1xp
    # model.add(keras.layers.Conv2D(filters=512, kernel_size=(filt_size, 1), padding='same', activation='relu',
    #                               input_shape=(p, 1, 1)))
    # if bn:
    #     model.add(BatchNormalization(axis=3))
    #     # model.add(MaxPool2D(pool_size=(p_size, 1), padding="same"))
    #     model.add(Dropout(do))
    # model.add(keras.layers.Conv2D(filters=256, kernel_size=(filt_size, 1), padding='same', activation='relu'))
    # if bn:
    #     model.add(BatchNormalization(axis=3))
    #     # model.add(MaxPool2D(pool_size=(p_size, 1), padding="same"))
    #     model.add(Dropout(do))
    #
    # model.add(keras.layers.Conv2D(filters=128, kernel_size=(filt_size, 1), padding='same', activation='relu'))
    # if bn:
    #     model.add(BatchNormalization(axis=3))
    #     # model.add(MaxPool2D(pool_size=(p_size, 1), padding="same"))
    #     model.add(Dropout(do))
    #
    # model.add(keras.layers.Conv2D(filters=64, kernel_size=(filt_size, 1), padding='same', activation='relu'))
    # if bn:
    #     model.add(BatchNormalization(axis=3))
    #     # model.add(MaxPool2D(pool_size=(p_size, 1), padding="same"))
    #     model.add(Dropout(do))
    #
    # model.add(keras.layers.Conv2D(filters=32, kernel_size=(filt_size, 1), padding='same', activation='relu'))
    # if bn:
    #     model.add(BatchNormalization(axis=3))
    #     # model.add(MaxPool2D(pool_size=(p_size, 1), padding="same"))
    #     model.add(Dropout(do))
    #
    # model.add(keras.layers.Conv2D(filters=16, kernel_size=(filt_size, 1), padding='same', activation='relu'))
    # if bn:
    #     model.add(BatchNormalization(axis=3))
    #     # model.add(MaxPool2D(pool_size=(p_size, 1), padding="same"))
    #     model.add(Dropout(do))
    #
    # model.add(keras.layers.Conv2D(filters=8, kernel_size=(filt_size, 1), padding='same', activation='relu'))
    # if bn:
    #     model.add(BatchNormalization(axis=3))
    #     # model.add(MaxPool2D(pool_size=(p_size, 1), padding="same"))
    #     model.add(Dropout(do))
    #
    # model.add(keras.layers.Conv2D(filters=4, kernel_size=(filt_size, 1), padding='same', activation='relu'))
    # if bn:
    #     model.add(BatchNormalization(axis=3))
    #     # model.add(MaxPool2D(pool_size=(p_size, 1), padding="same"))
    #     model.add(Dropout(do))
    #
    # model.add(keras.layers.Conv2D(filters=2, kernel_size=(filt_size, 1), padding='same', activation='relu'))
    # if bn:
    #     model.add(BatchNormalization(axis=3))
    #     # model.add(MaxPool2D(pool_size=(p_size, 1), padding="same"))
    #     model.add(Dropout(do))
    #
    # model.add(keras.layers.Flatten())
    # model.add(keras.layers.Dense(512, activation='relu'))
    # model.add(keras.layers.Dense(2, activation='linear'))





    # # conv layers 8xp
    # model.add(keras.layers.Conv2D(filters=64, kernel_size=(2, 2), padding='same', activation='relu',
    #                               input_shape=(8, p, 1)))
    # # model.add(keras.layers.Dropout(do))
    # # model.add(keras.layers.MaxPooling2D(pool_size=(2, 1)))
    # if bn:
    #     model.add(BatchNormalization())
    # model.add(keras.layers.Conv2D(filters=32, kernel_size=(2, 2), padding='same', activation='relu'))
    # # model.add(keras.layers.Dropout(do))
    # if bn:
    #     model.add(BatchNormalization())
    # model.add(keras.layers.Conv2D(filters=16, kernel_size=(2, 2), padding='same', activation='relu'))
    # # model.add(keras.layers.Dropout(do))
    # if bn:
    #     model.add(BatchNormalization())
    # model.add(keras.layers.Conv2D(filters=8, kernel_size=(2, 2), padding='same', activation='relu'))
    # # model.add(keras.layers.Dropout(do))
    # if bn:
    #     model.add(BatchNormalization())
    # model.add(keras.layers.Conv2D(filters=4, kernel_size=(2, 2), padding='same', activation='relu'))
    # model.add(keras.layers.Dropout(do))
    # if bn:
    #     model.add(BatchNormalization())
    #
    #
    # model.add(keras.layers.Flatten())
    # model.add(keras.layers.Dense(2, activation='linear'))


    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    model.summary()
    return model


# --------------------------
# train
def findBestHyperparameters(X_tr, ytr, X_te, yte):
    # define model hyperparameters
    nn = [64]
    l = [3]
    bn = [True]
    epochs = [40]
    bs = [601]
    do = [0.]

    # all combinations of hyperparameters
    H = np.array(np.meshgrid(nn, l, bn, epochs, bs, do)).T.reshape(-1, 6)
    # to find best performing hyperparams h*, initialize minimum val loss
    fCE_star = np.infty  # lowest final loss obtained by a hyperparam set
    model_star = None # weights trained by best hyperparam set
    j = 0  # current iteration in hyperparam set loop

    for h in H:
        # define this training sessions hyperparameters
        nn = int(h[0])
        l = int(h[1])
        bn = h[2]
        epochs = int(h[3])
        bs = int(h[4])
        do = h[5]

        # define model
        model = define_model(nn, l, bn, do)
        # Fit the model
        checkpoint = ModelCheckpoint('model_best_weights.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                     mode='min', period=1)
        history = model.fit(X_tr, ytr, validation_data=(X_te, yte), epochs=epochs, batch_size=bs, verbose=1,
                            callbacks=[checkpoint])
        plt.plot(history.history['loss'][10:])
        plt.plot(history.history['val_loss'][10:])
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Training Loss', 'Validation Loss'], loc='upper left')
        plt.grid(True)
        plt.show()

        fCE = history.history['val_loss'][-1]

        # check to see if this is the best performing hyperparam set
        if fCE < fCE_star:
            print('new best!')
            fCE_star = fCE
            model_star = model

        j += 1  # update counter
        print('(', j, '/', len(H), ')', 'final validation CE loss for h ', h, ': ', fCE)

    return model_star




# import training data set
# experimental
mypath = './grid_5_23_20/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
# test_idx = np.random.randint(0, len(onlyfiles))
# files = np.delete(onlyfiles, test_idx)

X_tr = []
Y_tr = []
vlines = [0]
cmap_lables = []
cmap_counter = 0
cmap_compare = []
for file in onlyfiles:
    mat = sio.loadmat(mypath+file)
    rss_data = mat['file'][0][0][0][0][0][0]
    idx = ~np.isnan(rss_data).any(axis=1)
    no_nan_rss_data = rss_data[idx]
    # col_mean = np.nanmean(rss_data, axis=0)*np.sqrt(2 / np.pi)
    # col_std = np.nanstd(rss_data, axis=0)
    # # Find indices that you need to replace
    # inds = np.where(np.isnan(rss_data))
    #
    # # replace NANs with random rayleigh values
    # for idx in range(len(inds[0])):
    #     if inds[1][idx] == 0:
    #         rss_data[inds[0][idx], inds[1][idx]] = np.random.rayleigh(col_mean[0], size=1)
    #     if inds[1][idx] == 1:
    #         rss_data[inds[0][idx], inds[1][idx]] = np.random.rayleigh(col_mean[1], size=1)
    #     else:
    #         rss_data[inds[0][idx], inds[1][idx]] = np.random.rayleigh(col_mean[2], size=1)
    #
    # # add even more simulated data points using column mean and variance such that each location has n samples
    # n = 1000 - len(rss_data)
    # # n=0
    # aug_data = np.random.rayleigh(col_mean, size=(n, 3))
    # rss_data = np.concatenate((rss_data, aug_data), axis=0)

    # compute packet RSS values
    num_packets = int(np.floor(len(no_nan_rss_data)/8.))
    packet_rss = []
    for i in range(num_packets):
        packet_rss.append(np.mean(no_nan_rss_data[i * 8:(i + 1) * 8], axis=0))
        cmap_lables.extend([cmap_counter])
        cmap_compare.extend([0])

    cmap_counter+= 1
    vlines.append(len(packet_rss))
    X_tr.extend(packet_rss)  # RSS values
    Y_tr.extend(np.tile(mat['file'][0][0][0][0][0][-1][0], reps=(len(packet_rss), 1)))  # tile stationary location labels

# pre-process data
X_tr = np.array(10*np.log10(X_tr))  # make into numpy arrays from native lists
Y_tr = np.array(Y_tr)










import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold

tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
X_tsne = tsne.fit_transform(X_tr.tolist())
plt.scatter(X_tsne[:,0],X_tsne[:,1],c=cmap_lables,cmap='viridis')
plt.title('t-SNE Dimension Reduction of Training Data')
plt.show()
















# # simulated
# X_sim = np.load('X_sim_grid.npy').T
# Y_sim = np.load('Y_sim_grid.npy').T
#
# # group packets
# X = []
# Y = []
# reps = 8
# for i in range(reps):
#     X.append(X_sim[i::reps])
#     Y.append(Y_sim[i::reps])
#
# X = np.array(X)
# Y = np.array(Y)
# X_sim = np.reshape(X, newshape=(X.shape[1], X.shape[0], X.shape[2]))
# Y_sim = np.reshape(Y, newshape=(Y.shape[1], Y.shape[0], Y.shape[2]))
# Y_sim = Y_sim[:,0,:]
# X_sim = np.mean(X_sim, axis=1)
#
# X_tr = np.concatenate((X_tr, X_sim),axis=0)
# Y_tr = np.concatenate((Y_tr, Y_sim),axis=0)





# import val/test data
mypath = './los_2_data_5_23._20/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
X = []
Y = []

from scipy import stats
cmap_lables=[]
vlines_test = [0]
cmap_counter = 0
p_rss_by_loc = []
loc_key = []
for file in onlyfiles:
    mat = sio.loadmat(mypath + file)
    rss_data = mat['file'][0][0][0][0][0][0]
    # remove rows with NANs
    idx = ~np.isnan(rss_data).any(axis=1)
    rss_data = rss_data[idx]

    # compute packet RSS values
    num_packets = int(np.floor(len(rss_data)/8.))
    packet_rss = []
    for i in range(num_packets):
        packet_rss.append(np.mean(rss_data[i*8:(i+1)*8], axis=0))
        cmap_lables.extend([cmap_counter])
        cmap_compare.extend([1])

    p_rss_by_loc.append(10*np.log10(packet_rss))
    loc_key.append(mat['file'][0][0][0][0][0][-1][0])

    cmap_counter += 1
    X.extend(packet_rss)  # RSS values
    vlines_test.append(len(packet_rss))
    Y.extend(np.tile(mat['file'][0][0][0][0][0][-1][0], reps=(len(packet_rss), 1)))  # tile stationary location labels



# file = onlyfiles[test_idx]
# mat = sio.loadmat(mypath + file)
# rss_data = mat['file'][0][0][0][0][0][0]
# idx = ~np.isnan(rss_data).any(axis=1)
# rss_data = rss_data[idx]
#
# # compute packet RSS values
# num_packets = int(np.floor(len(rss_data)/8.))
# packet_rss = []
# for i in range(num_packets):
#     packet_rss.append(np.mean(rss_data[i*8:(i+1)*8], axis=0))
#
# X = packet_rss  # RSS values
# Y = np.tile(mat['file'][0][0][0][0][0][-1][0], reps=(len(packet_rss), 1))  # tile stationary location labels
#
# pre-process data
X = np.array(10*np.log10(X))
Y = np.array(Y)



# WCL localization benchmark
# Estimated NLOS alpha:  -3.850275788138077
# Estimated LOS alpha:  -2.053987479176067
LOS_grad = -2.053987479176067
NLOS_grad = -3.850275788138077
avg_alpha = np.mean([LOS_grad, NLOS_grad])
avg_alpha = NLOS_grad
receivers = np.array([[7.5,0],[0,12.99],[15,12.99]])  # receiver locations, a_i in literature

g = 1.  # weight parameter
wcl_locs = []
dist_err = []
true_dists = []
est_dists = []
for j in range (len(X)):
    P0 = -55.8297023474628  # 1m reference power
    distance = np.power(10, (X[j] - P0)/(10. * avg_alpha))  # estimated distance to each receiver
    # print(distance)
    num_x, den_x, num_y, den_y = [], [], [], []
    for i in range(len(X[0])):
        # computations towards distance MSE
        true_dist = np.sqrt((Y[j,0] - receivers[i,0])**2 + (Y[j,1] - receivers[i,1])**2)
        true_dists.append(true_dist)
        est_dists.append(distance[i])
        dist_err.append((distance - true_dist)**2)
        # computations towards location estimates
        num_x.append((distance[i] ** (-g) * receivers[i,0]))
        den_x.append(distance[i] ** (-g))
        num_y.append((distance[i] ** (-g) * receivers[i,1]))
        den_y.append(distance[i] ** (-g))
    x_est = sum(num_x) / sum(den_x)
    y_est = sum(num_y) / sum(den_y)
    wcl_locs.append([x_est, y_est])

print('WCL Distance MSE: %f' % np.mean(dist_err))
wcl_locs = np.array(wcl_locs)
mse = np.mean((wcl_locs - Y)**2)
plt.scatter(wcl_locs[:,0], wcl_locs[:,1])
plt.scatter(Y[:,0],Y[:,1])
plt.scatter(receivers[:,0], receivers[:,1])
plt.legend(['WCL estimates','Truth','Receivers'])
plt.title('Distance MSE: %f, WCL Localization MSE: %f' % (np.mean(dist_err), mse))
plt.grid(True)
plt.xlabel('X-coordainte (meters)')
plt.ylabel('Y-coordainte (meters)')
plt.show()

# np.save('CNN_preds_normal',wcl_locs)
# np.save('CNN_true_normal',Y)


# for i in range(len(x_location)):
#     vehicle_rss = []
#     dist_est = []
#     for j in RSS.keys():
#         vehicle_rss.append(RSS[j][i])
#     top_3_index = sorted(range(len(vehicle_rss)), reverse=True, key=lambda k: vehicle_rss[k])[:3]
#
#     for top_3 in top_3_index:
#         dist_est.append(distance(vehicle_rss[top_3]))
#
#     sensor_names = []
#     for new in top_3_index:
#         sensor_names.append(RSS.keys()[new])
#
#     three_sensor_location = []
#     for l in sensor_names:
#         three_sensor_location.append(sensor_location[l])
#
#     num_x, den_x, num_y, den_y = [],[],[],[]
#     for loc in range(len(dist_est)):
#         d_est = dist_est[loc]
#         num_x.append((d_est**(-g)*three_sensor_location[loc][0]))
#         den_x.append(d_est**(-g))
#         num_y.append((d_est ** (-g)*three_sensor_location[loc][1]))
#         den_y.append(d_est ** (-g))
#     x_est = sum(num_x) / sum(den_x)
#     y_est = sum(num_y) / sum(den_y)
#     estimated_x_error.append(abs(x_location[i]-x_est))
#     estimated_y_error.append(abs(abs(y_location[i])-abs(y_est)))
#     d_true = (x_location[i], y_location[i])
#     d_est = (x_est, y_est)
#
#     dme_all.append(euclidean(d_true,d_est))
#
# a = sum(dme_all)/len(dme_all)
# print(a)
# DME.append(a)
# #print('Mean Distance Measurement Error:%f ' %a)




# KW test for non-parametric means
for rec in range(3):
    for i in range(len(p_rss_by_loc)):
        for j in range(i, len(p_rss_by_loc)):
            if i != j:
                arri = np.array(p_rss_by_loc[i])
                arrj = np.array(p_rss_by_loc[j])
                stat, p_stat = stats.kruskal(arri[:,rec], arrj[:,rec])

                if p_stat > 0.05:
                    print('p: ',p_stat)
                    print('receiver: ',rec,' statistically similar for locations: ')
                    print(loc_key[i])
                    print(loc_key[j])



tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
X_tsne = tsne.fit_transform(X.tolist())
plt.scatter(X_tsne[:,0],X_tsne[:,1],c=cmap_lables,cmap='plasma')
plt.title('t-SNE Dimension Reduction of Val/Test Data')
plt.show()


tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
X_tsne = tsne.fit_transform(np.concatenate((X_tr,X),axis=0))
plt.scatter(X_tsne[:,0],X_tsne[:,1],c=cmap_compare,cmap='bwr')
plt.title('t-SNE Dimension Reduction of Training and Val/Test Data')
plt.show()

plt.plot(np.concatenate((X_tr[:,0], X[:,0]),axis=0))
count = 0
for vline in vlines:
    plt.axvline(x=vline+count,linewidth=2, color='r')
    count += vline
for vline in vlines_test:
    plt.axvline(x=vline+count,linewidth=2, color='g')
    count += vline
plt.ylabel('Power (dBm)')
plt.xlabel('Sample #')
plt.title('Receiver 1 RSS By Location')
plt.show()


plt.plot(np.concatenate((X_tr[:,1], X[:,1]),axis=0))
count = 0
for vline in vlines:
    plt.axvline(x=vline+count,linewidth=2, color='r')
    count += vline
for vline in vlines_test:
    plt.axvline(x=vline+count,linewidth=2, color='g')
    count += vline
plt.ylabel('Power (dBm)')
plt.xlabel('Sample #')
plt.title('Receiver 2 RSS By Location')
plt.show()

plt.plot(np.concatenate((X_tr[:,2], X[:,2]),axis=0))
count = 0
for vline in vlines:
    plt.axvline(x=vline+count,linewidth=2, color='r')
    count += vline
for vline in vlines_test:
    plt.axvline(x=vline+count,linewidth=2, color='g')
    count += vline
plt.ylabel('Power (dBm)')
plt.xlabel('Sample #')
plt.title('Receiver 3 RSS By Location')
plt.show()


def return_intersection(hist_1, hist_2):
    minima = np.minimum(hist_1, hist_2)
    intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
    return intersection


bins = 100
raange = [-120, -70]
hist1, _ = np.histogram(X_tr[:,0].flatten(), bins=bins, range=raange)
hist2, _ = np.histogram(X[:,0].flatten(), bins=bins, range=raange)
intersection = return_intersection(hist1, hist2)
plt.hist([np.expand_dims(X_tr[:,0].flatten(),axis=1).tolist(), np.expand_dims(X[:,0].flatten(),axis=1).tolist()], bins=bins, density=True, alpha=0.5)
plt.xlabel('Power (dBm)')
plt.ylabel('density')
plt.legend(['Training Data','Testing Data'])
plt.title('Receiver 1 RSS Histogram Intersect: % f' % intersection)
plt.show()

hist1, _ = np.histogram(X_tr[:,1].flatten(), bins=bins, range=raange)
hist2, _ = np.histogram(X[:,1].flatten(), bins=bins, range=raange)
intersection = return_intersection(hist1, hist2)
plt.hist([np.expand_dims(X_tr[:,1].flatten(),axis=1).tolist(), np.expand_dims(X[:,1].flatten(),axis=1).tolist()], bins=bins, density=True, alpha=0.5)
plt.xlabel('Power (dBm)')
plt.ylabel('density')
plt.legend(['Training Data','Testing Data'])
plt.title('Receiver 2 RSS Histogram Intersect: % f' % intersection)
plt.show()

hist1, _ = np.histogram(X_tr[:,2].flatten(), bins=bins, range=raange)
hist2, _ = np.histogram(X[:,2].flatten(), bins=bins, range=raange)
intersection = return_intersection(hist1, hist2)
plt.hist([np.expand_dims(X_tr[:,2].flatten(),axis=1).tolist(), np.expand_dims(X[:,2].flatten(),axis=1).tolist()], bins=bins, density=True, alpha=0.5)
plt.xlabel('Power (dBm)')
plt.ylabel('density')
plt.legend(['Training Data','Testing Data'])
plt.title('Receiver 3 RSS Histogram Intersect: % f' % intersection)
plt.show()





# X = np.load('X_sim_los.npy').T
# Y = np.load('Y_sim_los.npy').T
#
# # group packets
# X_temp = []
# Y_temp = []
#
# for i in range(reps):
#     X_temp.append(X[i::reps])
#     Y_temp.append(Y[i::reps])
#
# X_temp = np.array(X_temp)
# Y_temp = np.array(Y_temp)
# X = np.reshape(X_temp, newshape=(X_temp.shape[1], X_temp.shape[0], X_temp.shape[2]))
# Y = np.reshape(Y_temp, newshape=(Y_temp.shape[1], Y_temp.shape[0], Y_temp.shape[2]))
# Y = Y[:,0,:]
# X = np.mean(X, axis=1)
#
# print(X.shape)
# print(Y.shape)
# print(Y[0])



# np.random.seed(seed=711)  # shuffle to de-group labels
# np.random.shuffle(X)
# np.random.seed(seed=711)
# np.random.shuffle(Y)

# # change labels to estimate ranges instead of locations
# Y_train = []
# for i in range(len(Y_tr)):
#     Y_train.append(np.linalg.norm(Y_tr[i] - receivers, axis=1))
# Y_train = np.array(Y_train)
# Y_tr = Y_train
#
# Y_locs = Y  # save original labels for later
# Y_ = []
# for i in range(len(Y)):
#     Y_.append(np.linalg.norm(Y[i] - receivers, axis=1))
# Y_ = np.array(Y_)
# Y = Y_
#
# print(Y_tr.shape)
# print(Y.shape)

np.random.seed(seed=711)  # shuffle to de-group labels
np.random.shuffle(X_tr)
np.random.seed(seed=711)
np.random.shuffle(Y_tr)


np.random.seed(seed=711)  # shuffle to de-group labels
np.random.shuffle(X)
np.random.seed(seed=711)
np.random.shuffle(Y)


X_tr = np.expand_dims(np.expand_dims(X_tr, axis=-1), axis=-1)
X = np.expand_dims(np.expand_dims(X, axis=-1), axis=-1)

# X_tr = np.expand_dims(X_tr, axis=-1)
# X = np.expand_dims(X, axis=-1)


# print('training data shape: ',X_tr.shape)
# print('test data shape: ',X.shape)
# # partition
# X_val = X[0:int(len(X)/2.)]
# Y_val = Y[0:int(len(X)/2.)]

X_val = X
Y_val = Y

# Y = Y_tr
# X_tr = X[:-200]
# Y_tr = Y[:-200]
#
# X_val = X[-200:]
# Y_val = Y[-200:]

# train
model = findBestHyperparameters(X_tr, Y_tr, X_val, Y_val)
# model.save('saved_CNN.h5')

# X_te = X[int(len(X)/2.):]
# Y_te = Y[int(len(X)/2.):]

X_te = X
Y_te = Y

model = keras.models.load_model('model_best_weights.h5')


# TEST
score = model.evaluate(X_te, Y_te, verbose=0)  # Print test accuracy
print('\n', 'Test MSE:', score[1])

# # WCL with ranges estimated by CNN
#
# wcl_locs = []
# dist_err = []
# true_dists = []
# est_dists = []
# distances = model.predict(X_te)  # estimated distance to each receiver
# g = 8.
# for j in range(len(X_te)):
#     distance = distances[j]
#     # print(distance)
#     num_x, den_x, num_y, den_y = [], [], [], []
#     for i in range(len(X[0])):
#         # computations towards distance MSE
#         true_dist = np.sqrt((Y_locs[j,0] - receivers[i,0])**2 + (Y_locs[j,1] - receivers[i,1])**2)
#         true_dists.append(true_dist)
#         est_dists.append(distance[i])
#         dist_err.append((distance - true_dist)**2)
#         # computations towards location estimates
#         num_x.append((distance[i] ** (-g) * receivers[i,0]))
#         den_x.append(distance[i] ** (-g))
#         num_y.append((distance[i] ** (-g) * receivers[i,1]))
#         den_y.append(distance[i] ** (-g))
#     x_est = sum(num_x) / sum(den_x)
#     y_est = sum(num_y) / sum(den_y)
#     wcl_locs.append([x_est, y_est])

# print('CNN/WCL Distance MSE: %f' % np.mean(dist_err))
# wcl_locs = np.array(wcl_locs)
# mse = np.mean((wcl_locs - Y_locs)**2)
# plt.scatter(wcl_locs[:,0], wcl_locs[:,1])
# plt.scatter(Y_locs[:,0],Y_locs[:,1])
# plt.scatter(receivers[:,0], receivers[:,1])
# plt.legend(['WCL estimates','Truth','Receivers'])
# plt.title('Distance MSE: %f, WCL Localization MSE: %f' % (np.mean(dist_err), mse))
# plt.grid(True)
# plt.xlabel('X-coordainte (meters)')
# plt.ylabel('Y-coordainte (meters)')
# plt.show()

plt.scatter(Y_tr[:,0], Y_tr[:,1])
plt.scatter(Y_te[:,0], Y_te[:,1])
preds = model.predict(X_te)
plt.scatter(preds[:,0],preds[:,1])
plt.scatter(receivers[:,0], receivers[:,1])
plt.legend(['Training','Test Truth','Test Estimates','Receivers'])
plt.grid(True)
plt.xlabel('X-coordainte (meters)')
plt.ylabel('Y-coordainte (meters)')
plt.title('Regression MSE: %s' % str(score[1]))
plt.show()


preds_train = model.predict(X_tr)

raange = [0, 15]
hist1, _ = np.histogram(preds_train[:,0].flatten(), bins=bins, range=raange)
hist2, _ = np.histogram(preds[:,0].flatten(), bins=bins, range=raange)
intersection = return_intersection(hist1, hist2)
plt.hist([np.expand_dims(preds_train[:,0].flatten(),axis=1).tolist(), np.expand_dims(preds[:,0].flatten(),axis=1).tolist()], bins=bins, density=True, alpha=0.5)
plt.xlabel('Position (m)')
plt.ylabel('density')
plt.legend(['Training Data','Testing Data'])
plt.title('Predicted X-coordinate Histogram Intersect: % f' % intersection)
plt.show()

hist1, _ = np.histogram(preds_train[:,1].flatten(), bins=bins, range=raange)
hist2, _ = np.histogram(preds[:,1].flatten(), bins=bins, range=raange)
intersection = return_intersection(hist1, hist2)
plt.hist([np.expand_dims(preds_train[:,1].flatten(),axis=1).tolist(), np.expand_dims(preds[:,1].flatten(),axis=1).tolist()], bins=bins, density=True, alpha=0.5)
plt.xlabel('Position (m)')
plt.ylabel('density')
plt.legend(['Training Data','Testing Data'])
plt.title('Predicted Y-coordinate Histogram Intersect: % f' % intersection)
plt.show()

np.save('CNN_preds_normal',preds)
np.save('CNN_true_normal',Y_te)


