import numpy as np
import matplotlib.pyplot as plt


n=10

a = np.random.multivariate_normal(mean=(7,7), cov=[[1,-.5],[-.5,1]], size=n)
b = np.random.multivariate_normal(mean=(9,10), cov=[[1,0],[0,1]], size=n)
c = np.random.multivariate_normal(mean=(8,7), cov=[[2,0],[0,1]], size=n)

fig, ax = plt.subplots()
ax.scatter(np.concatenate((a[:,0],b[:,0],c[:,0]), axis=0), np.concatenate((a[:,1],b[:,1],c[:,1]), axis=0))
# ax.scatter(a[:,0], a[:,1])
# ax.scatter(b[:,0], b[:,1])
# ax.scatter(c[:,0], c[:,1])



fig.patch.set_visible(False)
ax.axis('off')
plt.show()

