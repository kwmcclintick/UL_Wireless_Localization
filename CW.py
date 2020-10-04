from chinese_whispers import __version__ as cw_version
from networkx import __version__ as nx_version
print('Chinese Whispers {0}'.format(cw_version))
print('NetworkX {0}'.format(nx_version))

from itertools import product
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from chinese_whispers import chinese_whispers, aggregate_clusters




means = np.load('DPGMM_means.npy')

G = nx.Graph()
for i in range(len(means)):
    G.add_node(i, pos=means[i])

weights = []
for i in range(len(means)):
    for j in range(len(means)):
        if i != j:
            # assign weights. Large weights (small distances) get clustered together, and score is proportional to number
            # of nodes
            G.add_edge(i,j,weight=-np.linalg.norm(means[i]-means[j], ord=2)**len(means))
            print(G.edges()[i,j])
            weights.append(G.edges()[i,j]['weight'])


# G.add_edges_from((a,b) for a,b in product(range(len(means)), range(len(means))) if a != b)

chinese_whispers(G, weighting='top')

# Print the clusters in the descending order of size
print('Cluster ID\tCluster Elements\n')
cluster_mean = []
for label, cluster in sorted(aggregate_clusters(G).items(), key=lambda e: len(e[1]), reverse=True):
    # print('{}\t{}\n'.format(label, cluster))
    idxs = list(cluster)
    ms = []
    for idx in idxs:
        node_data = G.nodes().data()[idx]
        ms.append(node_data['pos'].tolist())

    ms = np.array(ms)
    test = np.mean(ms, axis=0)
    cluster_mean.append(test)

    plt.scatter(ms[:,0], ms[:,1], marker='^', s=100, label='Tx C'+str(label))

mean_G = nx.Graph()
for i in range(len(cluster_mean)):
    mean_G.add_node(i, pos=cluster_mean[i])

cluster_mean = np.array(cluster_mean)
truth = np.array([[5.9715, 6.7265], [5.3495, 9.2845]])
center_mse = np.mean((cluster_mean - truth)**2)
print(center_mse)
colors = [1. / G.nodes[node]['label'] for node in G.nodes()]
pos=nx.get_node_attributes(G,'pos')
plt.scatter(cluster_mean[:,0], cluster_mean[:,1], marker='x', s=100, label='Means')
plt.scatter(truth[:,0], truth[:,1], marker='x', s=100, label='Truths')
plt.title('Vehicle Centroid MSE: %f' % center_mse)
plt.legend()
plt.grid(True)
plt.xlim([2, 13])
plt.ylim([2, 13])
# nx.draw_networkx(G, pos, cmap=plt.get_cmap('jet'), node_color=colors, font_color='white')
plt.show()



