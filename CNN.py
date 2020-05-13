"""


"""


import math
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense
from keras.models import Sequential
from keras import optimizers
import matplotlib.pyplot as plt
import keras
from sklearn import preprocessing

p = 3


def define_model(nn, l, alpha):
    model = keras.Sequential()
    # conv layers
    model.add(keras.layers.Conv2D(filters=nn, kernel_size=(2, 1), padding='same', activation='relu',
                                  input_shape=(p, 1, 1)))
    for _ in range(l-1):
        model.add(keras.layers.Conv2D(filters=nn, kernel_size=(2, 1), padding='same', activation='relu'))
        # model.add(keras.layers.MaxPooling2D(pool_size=(2, 1)))
        model.add(keras.layers.Dropout(0.3))
    # # dense layers
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(2, activation='linear'))
    sgd = keras.optimizers.SGD(lr=alpha)
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    model.summary()
    return model


# --------------------------
# train
def findBestHyperparameters(X_tr, ytr, X_te, yte):
    # define model hyperparameters
    nn = [5]
    l = [2]
    alpha = [1e-2]
    epochs = [50]
    bs = [5]

    # all combinations of hyperparameters
    H = np.array(np.meshgrid(nn, l, alpha, epochs, bs)).T.reshape(-1, 5)
    # to find best performing hyperparams h*, initialize minimum val loss
    fCE_star = np.infty  # lowest final loss obtained by a hyperparam set
    model_star = None # weights trained by best hyperparam set
    j = 0  # current iteration in hyperparam set loop

    for h in H:
        # define this training sessions hyperparameters
        nn = int(h[0])
        l = int(h[1])
        alpha = h[2]
        epochs = int(h[3])
        bs = int(h[4])

        # define model
        model = define_model(nn, l, alpha)
        # train
        history = model.fit(X_tr, ytr, validation_data=(X_te, yte), epochs=epochs, batch_size=bs, verbose=0)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
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
X = np.expand_dims(np.expand_dims(preprocessing.scale(np.load('X.npy').T), axis=-1), axis=-1)
Y = np.load('Y.npy').T

print(X.shape)

# shuffle in groups of 4 tires and 4 cars (16) for time snapshots
X = np.reshape(X, newshape=(1000,8,p,1,1))
Y = np.reshape(Y, newshape=(1000,8,2))
np.random.seed(seed=711)
np.random.shuffle(X)
np.random.seed(seed=711)
np.random.shuffle(Y)

# ungroup again
X = np.reshape(X, newshape=(8000,p,1,1))
Y = np.reshape(Y, newshape=(8000,2))


# import and append novel training data
# X_gan = np.load('novel_images.npy').astype(np.float64)
# y_gan = np.load('novel_labels.npy').astype(np.float64)

# X_tr = np.append(X_tr, X_gan, axis=0)
# y_tr = np.append(y_tr, y_gan, axis=0)



# partition
X_tr = X[0:7000]
Y_tr = Y[0:7000]
X_val = X[7000:7520]
Y_val = Y[7000:7520]

# train
model = findBestHyperparameters(X_tr, Y_tr, X_val, Y_val)
model.save('saved_model.h5')


X_te = X[7520:]
Y_te = Y[7520:]
# TEST
score = model.evaluate(X_te, Y_te, verbose=0)  # Print test accuracy
print('\n', 'Test MSE:', score[1])


plt.scatter(Y_te[:,0], Y_te[:,1])
preds = model.predict(X_te)
plt.scatter(preds[:,0],preds[:,1])
plt.legend(['Truth', 'Estimates'])
plt.grid(True)
plt.xlabel('X-coordainte (meters)')
plt.ylabel('Y-coordainte (meters)')
plt.xlim([0, 40])
plt.ylim([-2,6])
plt.show()


np.save('CNN_preds',preds)
np.save('CNN_true',Y_te)


