import random

import matplotlib.pyplot as plt
import networkx as nx

M = 1000
T = 100
K = 3
N = 30


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
        self.free = sorted([x for x in sorted(self.graph.nodes) if x not in self.terminaux])
        self.population = []

    def fitness(self, chaine):
        if len(self.free) != len(chaine):
            raise Exception('Codedage invalide')
        new = nx.Graph()
        weight = 0
        new.add_nodes_from(self.terminaux)
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
        dict = {}
        for node in self.terminaux:
            dict[node] = 1
        values = [dict.get(node, 0.5) for node in self.graph.nodes()]
        nx.draw_networkx(self.graph, with_labels=True, node_color=values)
        plt.show()

    def population_sorted(self, population):
        pop = []
        for individu in population:
            _, w, c = self.fitness(individu)
            if c:
                pop.append((individu, w))
        return sorted(pop, key=lambda x: x[1])

    def generate(self, num_individu, p):  # 0.2 < p < 0.5
        population = []
        for i in range(num_individu):
            individu = []
            for j in range(len(self.free)):
                individu.append(1) if random.random() < p else individu.append(0)
            if individu not in population:
                population.append(individu)
        self.population = self.population_sorted(population)
        if len(self.population) == 0: #pour etre sur d'avoir une solution realisable
            self.population = self.population_sorted([[1 for i in range(len(self.free))]])
        return self.population

    def new_generation(self, parent=False, type=False):
        old_pop = []
        best_score = self.population[0][1]
        for i in self.population:
            if not parent: # meilleurs parents
                if i[1] <= self.population[0][1]:
                    old_pop.append(i)
            else:
                note = float(best_score)/i[1]
                if random.random() < note:
                    old_pop.append(i)
        new_pop = []
        if len(old_pop) == 1:
            for i in range(T):
                m = mutation(old_pop[0][0])
                if m not in new_pop:
                    new_pop.append(m)
        else:
            for i in range(len(old_pop)):
                for j in range(i + 1, len(old_pop)):
                    child = croisement_point(old_pop[i][0], old_pop[j][0])
                    if child not in new_pop:
                        new_pop.append(child)
        if not type: # generational
            self.population = self.population_sorted(new_pop)
        else:  # elitist
            new_pop_score = self.population_sorted(new_pop)
            best_score = min(self.population[0][1], new_pop_score[0][1]) if len(new_pop_score) > 0 else self.population[0][1]
            final_pop = []
            for i in self.population:
                if random.random() < float(best_score)/i[1]:
                    if i[0] not in final_pop:
                        final_pop.append(i[0])
            for i in new_pop_score:
                if random.random() < float(best_score)/i[1]:
                    if i[0] not in final_pop:
                        final_pop.append(i[0])
            self.population = self.population_sorted(final_pop)
            '''
            self.population = []
            for i in sort:
                if i[1] <= sort[0][1]:
                    if i not in self.population:
                        self.population.append(i)'''
        return self.population

    def heuristic(self):
        old_pop = self.population
        print(old_pop)
        old_score = old_pop[0][1]
        new_pop = self.new_generation(parent=True,type=True)
        print(new_pop)
        new_score = new_pop[0][1]
        while old_score == new_score :
            old_score = new_score
            for i in range(K):
                new_pop = self.new_generation(parent=True, type=True)
                new_score = new_pop[0][1]
                print(new_pop)
            if old_score == new_score:
                break
        while old_score != new_score:
            old_score = new_score
            for i in range(K):
                new_pop = self.new_generation(parent=True,type=True)
                new_score = new_pop[0][1]
                print(new_pop)
        self.draw_individu(new_pop[0][0])

    def draw_individu(self,individu):
        graph, _, _ = self.fitness(individu)
        dict = {}
        for node in self.terminaux:
            dict[node] = 1
        values = [dict.get(node, 0.5) for node in graph.nodes()]
        nx.draw_networkx(graph, with_labels=True, node_color=values)
        plt.show()


def mutation(chaine):
    new = []
    for char in chaine:
        if random.random() < 0.05:
            new.append((char + 1) % 2)
        else:
            new.append(char)
    return new


def croisement_point(parent1, parent2):
    p_rdv = int(len(parent1)/2)
    enfant = []
    for i in range(p_rdv):
        enfant.append(parent1[i])
    for i in range(p_rdv, len(parent2)):
        enfant.append(parent2[i])
    return enfant


g, t = readfile("B/b05.stp")
ex = SteinerGraphe(g, t)
#ex.draw()
# ex.draw()
# a,b,_=ex.fitness([1, 1, 0, 0])

ex.generate(N, 0.5)
ex.heuristic()