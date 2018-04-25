import random

import matplotlib.pyplot as plt
import networkx as nx

M = 1000


def readfile(filename):
    G = nx.Graph()
    T = []
    file = open(filename, "r")
    for line in file.readlines():
        s = line.split()
        if len(s) != 0:
            if s[0] == "E":
                G.add_edge(s[1], s[2], weight=int(s[3]))
            if s[0] == "T":
                T.append(s[1])
    file.close()
    return G, T


class SteinerGraphe:
    def __init__(self, graph: nx.Graph, terminaux: list):
        self.graph = graph
        self.terminaux = sorted(terminaux)
        self.free = [x for x in sorted(self.graph.nodes) if x not in self.terminaux]
        self.population = []

    def fitness(self, chaine):
        if len(self.free) != len(chaine):
            raise Exception('Codedage invalide')
        new = nx.Graph()
        weight = 0
        # kruskal procedure
        for a, b, data in sorted(self.graph.edges(data=True), key=lambda x: x[2]['weight']):
            # print (a,b,data['weight'])
            if not ((a in self.free and chaine[self.free.index(a)] == 0) or (
                    b in self.free and chaine[self.free.index(b)] == 0)):
                new.add_edge(a, b, weight=data['weight'])
                weight += data['weight']
                if nx.algorithms.cycles.cycle_basis(new):
                    new.remove_edge(a, b)
                    weight -= data['weight']
        # print(weight)
        # nx.draw_networkx(new, with_labels=True)
        # plt.show()
        if nx.algorithms.components.is_connected(new):  # connexe
            return new, weight, True
        else:
            return new, weight + M * (new.number_of_nodes() - 1 - new.number_of_edges()), False

    def draw(self):
        nx.draw_networkx(self.graph, with_labels=True)
        plt.show()

    def population_sorted(self, population):
        pop = []
        for individu in population:
            _, w, _ = self.fitness(individu)
            pop.append((individu, w))
        self.population = sorted(pop, key=lambda x: x[1])
        return self.population

    def generate(self, num_individu, p):  # 0.2 < p < 0.5
        population = []
        for i in range(num_individu):
            individu = []
            for j in range(len(self.free)):
                individu.append(1) if random.random() < p else individu.append(0)
            population.append(individu)
        self.population = population
        return population


# TODO: 1.3 non traité !

def mutation(chaine):
    new = []
    for char in chaine:
        if random.random() < 0.05:
            new.append((char + 1) % 2)
        else:
            new.append(char)
    return new


def croisement_point(parent1, parent2, p_rdv):
    if p_rdv > len(parent1):
        raise Exception('Point hors base')
    enfant = []
    for i in range(p_rdv):
        enfant.append(parent1[i])
    for i in range(p_rdv, len(parent2)):
        enfant.append(parent2[i])
    return enfant


g, t = readfile("test.gr")
ex = SteinerGraphe(g, t)
# ex.draw()
# a,b,_=ex.fitness([1, 1, 0, 0])

gen = ex.generate(4, 0.5)
print(ex.population_sorted(gen))

# print(croisement_point([0,1,1,0,0],[0,0,0,1,1],3))
