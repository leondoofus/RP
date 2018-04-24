import random

import matplotlib.pyplot as plt
import networkx as nx

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


# TODO: 1.3 non traité !

def mutation(chaine):
    new = []
    for char in chaine:
        if random.random() < 0.05:
            new.append((char + 1) % 2)
        else:
            new.append(char)
    return new


def generate(numIndividu, length, p): #p entre 0.2 et 0.5
    population = []
    for i in range(numIndividu):
        individu = []
        for j in range(length):
            individu.append(1) if random.random() < p else individu.append(0)
        population.append(individu)
    return population


g, t = readfile("test.gr")
print(g.nodes)
print(t)
nx.draw_networkx(g, with_labels=True)
plt.show()
print(fitness(g, t, [1, 1, 0, 0]))
