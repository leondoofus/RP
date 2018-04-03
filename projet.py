import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
V = ['v1','v2','v3','v4','v5','u1','u2','u3','u4']
T = ['v1','v2','v3','v4','v5']
G.add_nodes_from(V)
G.add_edge('v1','v2',weight=8)
G.add_edge('v1','v3',weight=9)
G.add_edge('v1','u1',weight=2)
G.add_edge('v1','u3',weight=2)
G.add_edge('v2','v5',weight=5)
G.add_edge('v2','u1',weight=2)
G.add_edge('v2','u4',weight=8)
G.add_edge('v3','v4',weight=8)
G.add_edge('v3','u2',weight=4)
G.add_edge('v3','u3',weight=8)
G.add_edge('v4','u2',weight=3)
G.add_edge('v5','u2',weight=5)
G.add_edge('v5','u4',weight=8)
G.add_edge('u1','u2',weight=1)

nx.draw_networkx(G, with_labels=True)
plt.show()