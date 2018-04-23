import networkx as nx
import matplotlib.pyplot as plt
from readfile import readfile

M = 1000


def fitness(g, t, chaine):
    if len(g.nodes) - len(t) != len(chaine):
        raise Exception('Codedage invalide')
    l = [x for x in sorted(g.nodes) if x not in t]
    # print(l)
    new = nx.Graph()
    weight = 0
    for a, b, data in sorted(g.edges(data=True), key=lambda x: x[2]['weight']):
        # print (a,b,data['weight'])
        if not ((a in l and chaine[l.index(a)] == 0) or (b in l and chaine[l.index(b)] == 0)):
            new.add_edge(a, b, weight=data['weight'])
            weight += data['weight']
            if nx.algorithms.cycles.cycle_basis(new):
                new.remove_edge(a, b)
                weight -= data['weight']
    # print(weight)
    nx.draw_networkx(new, with_labels=True)
    plt.show()
    if nx.algorithms.components.is_connected(new):  # connexe
        return new, weight, True
    else:
        return new, weight + M * (new.number_of_nodes() - 1 - new.number_of_edges()), False


g, t = readfile("test.gr")
print(g.nodes)
print(t)
nx.draw_networkx(g, with_labels=True)
plt.show()
print(fitness(g, t, [1, 1, 0, 0]))
