from chinese_whispers import __version__ as cw_version
from networkx import __version__ as nx_version
print('Chinese Whispers {0}'.format(cw_version))
print('NetworkX {0}'.format(nx_version))

from itertools import product
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from chinese_whispers import chinese_whispers, aggregate_clusters

G = nx.karate_club_graph()

# Perform clustering of G, parameters weighting and seed can be omitted
chinese_whispers(G, weighting='top', seed=1337)

# Print the clusters in the descending order of size
print('Cluster ID\tCluster Elements\n')

for label, cluster in sorted(aggregate_clusters(G).items(), key=lambda e: len(e[1]), reverse=True):
    print('{}\t{}\n'.format(label, cluster))

colors = [1. / G.nodes[node]['label'] for node in G.nodes()]

nx.draw_networkx(G, cmap=plt.get_cmap('jet'), node_color=colors, font_color='white')
plt.title('Social Network Graph')
plt.show()


G = nx.Graph()
for i in range(10):
    G.add_node(i, pos=np.random.normal(0,1,size=(2,)))

for i in range(10,20):
    G.add_node(i, pos=np.random.normal(5,1,size=(2,)))

G.add_edges_from((a,b) for a,b in product(range(20), range(20)) if a != b)

chinese_whispers(G, weighting='top', seed=1337)

# Print the clusters in the descending order of size
print('Cluster ID\tCluster Elements\n')

for label, cluster in sorted(aggregate_clusters(G).items(), key=lambda e: len(e[1]), reverse=True):
    print('{}\t{}\n'.format(label, cluster))

colors = [1. / G.nodes[node]['label'] for node in G.nodes()]
pos=nx.get_node_attributes(G,'pos')
nx.draw_networkx(G, pos, cmap=plt.get_cmap('jet'), node_color=colors, font_color='white')
plt.title('Gaussian Cluster Graph')
plt.show()