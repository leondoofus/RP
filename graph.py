import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import time

M = 1000
T = 100
K = 1
N = 10
P = 0.5  # 0.2 < p < 0.5


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


def mutation(chaine):
    new = []
    for char in chaine:
        if random.random() < 0.05:
            new.append((char + 1) % 2)
        else:
            new.append(char)
    return new


def croisement_point(parent1, parent2):
    p_rdv = int(len(parent1) / 2)
    enfant = []
    for i in range(p_rdv):
        enfant.append(parent1[i])
    for i in range(p_rdv, len(parent2)):
        enfant.append(parent2[i])
    return enfant


class SteinerGraphe:
    def __init__(self, graph: nx.Graph, terminaux: list):
        self.graph = graph
        self.terminaux = sorted(terminaux)
        self.free = sorted([x for x in sorted(self.graph.nodes) if x not in self.terminaux])
        self.population = []

    def weight(self, graph):
        weight = 0
        for _, _, data in graph.edges(data=True):
            weight += data['weight']
        return weight

    def couvrant(self, graph):
        new = nx.minimum_spanning_tree(graph, 'weight')
        return new

    # Algorithme genetique
    # 1.2 Fitness
    def fitness(self, chaine):
        if len(self.free) != len(chaine):
            print(self.free)
            print(chaine)
            raise Exception('Codedage invalide')
        # Kruskal procedure
        graph = self.graph.copy()
        for i in range(len(self.free)):
            if chaine[i] == 0:
                graph.remove_node(self.free[i])
        self.couvrant(graph)
        if nx.algorithms.components.is_connected(graph):  # connexe
            return graph, self.weight(graph), True
        else:
            return graph, self.weight(graph) + M * (graph.number_of_nodes() - 1 - graph.number_of_edges()), False

    def draw(self):
        dict = {}
        for node in self.terminaux:
            dict[node] = 1
        values = [dict.get(node, 0.5) for node in self.graph.nodes()]
        nx.draw_networkx(self.graph, with_labels=True, node_color=values)
        plt.show()

    def draw_sousgraph(self, graph):
        dict = {}
        for node in self.terminaux:
            dict[node] = 1
        values = [dict.get(node, 0.5) for node in graph.nodes()]
        nx.draw_networkx(graph, with_labels=True, node_color=values)
        plt.show()

    def population_sorted(self, population):
        pop = []
        for individu in population:
            _, w, c = self.fitness(individu)
            if c:
                pop.append((individu, w))
        return sorted(pop, key=lambda x: x[1])

    def generate(self):
        population = []
        n = min(N, pow(2, len(self.free)))
        for i in range(n):
            individu = []
            for j in range(len(self.free)):
                individu.append(1) if random.random() < P else individu.append(0)
            while individu in population:
                individu = []
                for j in range(len(self.free)):
                    individu.append(1) if random.random() < P else individu.append(0)
            population.append(individu)
        self.population = self.population_sorted(population)
        if len(self.population) == 0:  # pour etre sur d'avoir une solution realisable
            self.population = self.population_sorted([[1 for i in range(len(self.free))]])
        return self.population

    def new_generation(self, parent=False, type=False):
        """
        :param parent: True for best parents, False for selection by probability
        :param type: True for Elitist, False for Generational
        """
        old_pop = []
        best_score = self.population[0][1]
        for i in self.population:
            if not parent:  # meilleurs parents
                if i[1] <= self.population[0][1]:
                    old_pop.append(i[0])
            else:
                note = float(best_score) / i[1]
                if random.random() < note:
                    old_pop.append(i[0])
        new_pop = []
        if len(old_pop) == 1:
            for i in range(T):
                m = mutation(old_pop[0])
                if m not in new_pop:
                    new_pop.append(m)
        else:
            for i in range(len(old_pop)):
                for j in range(i + 1, len(old_pop)):
                    child = croisement_point(old_pop[i], old_pop[j])
                    if child not in new_pop:
                        new_pop.append(child)
        if not type:  # generational
            self.population = self.population_sorted(new_pop)
            if len(self.population) == 0:
                self.population = self.population_sorted(old_pop)
        else:  # elitist
            new_pop_score = self.population_sorted(new_pop)
            best_score = min(self.population[0][1], new_pop_score[0][1]) if len(new_pop_score) > 0 else \
                self.population[0][1]
            final_pop = []
            for i in self.population:
                if random.random() < float(best_score) / i[1]:
                    if i[0] not in final_pop:
                        final_pop.append(i[0])
            for i in new_pop_score:
                if random.random() < float(best_score) / i[1]:
                    if i[0] not in final_pop:
                        final_pop.append(i[0])
            self.population = self.population_sorted(final_pop)
            if len(self.population) == 0:
                self.population = self.population_sorted(old_pop)
        return self.population

    def algorithm_genetic(self, parent=False, type=False):
        self.generate()
        old_pop = self.population
        print(old_pop)
        old_score = old_pop[0][1]
        new_pop = self.new_generation(parent=parent, type=type)
        print(new_pop)
        new_score = new_pop[0][1]
        while old_score == new_score:
            old_score = new_score
            for i in range(K):
                new_pop = self.new_generation(parent=parent, type=type)
                new_score = new_pop[0][1]
                # print(new_pop)
            if old_score == new_score:
                break
        while old_score != new_score:
            old_score = new_score
            for i in range(K):
                new_pop = self.new_generation(parent=parent, type=type)
                new_score = new_pop[0][1]
        self.draw_individu(new_pop[0][0])
        return new_pop[0]

    def draw_individu(self, individu):
        graph, _, _ = self.fitness(individu)
        dict = {}
        for node in self.terminaux:
            dict[node] = 1
        values = [dict.get(node, 0.5) for node in graph.nodes()]
        nx.draw_networkx(graph, with_labels=True, node_color=values)
        plt.show()

    # Heuristique du plus court chemin
    def terminaux_complets(self):  # 2.1.1
        new = nx.Graph()
        new.add_nodes_from(self.terminaux)
        for i in range(len(self.terminaux)):
            for j in range(i + 1, len(self.terminaux)):
                if self.graph.has_edge(self.terminaux[i], self.terminaux[j]):
                    new.add_edge(self.terminaux[i], self.terminaux[j],
                                 weight=self.graph.get_edge_data(self.terminaux[i], self.terminaux[j])['weight'])
                else:
                    w, _ = nx.single_source_dijkstra(self.graph, self.terminaux[i], target=self.terminaux[j],
                                                     weight='weight')
                    new.add_edge(self.terminaux[i], self.terminaux[j], weight=w)
        return new

    def remplacement(self, couvrant):  # 2.1.3
        new = nx.Graph()
        for a, b, data in sorted(couvrant.edges(data=True), key=lambda x: x[2]['weight']):
            if self.graph.has_edge(a, b):
                new.add_edge(a, b, weight=data['weight'])
            else:
                _, path = nx.single_source_dijkstra(self.graph, a, target=b, weight='weight')
                for i in range(len(path) - 1):
                    new.add_edge(path[i], path[i + 1], weight=self.graph.get_edge_data(path[i], path[i + 1])['weight'])
        # self.draw_sousgraph(new)
        return new

    def eliminate(self, graph):  # 2.1.5
        for node in graph.nodes:
            if node in self.free and graph.degree(node) == 1:
                graph.remove_node(node)
        # self.draw_sousgraph(graph)
        return graph, self.weight(graph)

    def heuristic_PCC(self):
        return self.eliminate(self.couvrant(self.remplacement(self.couvrant(self.terminaux_complets()))))

    # Heuristique de l’arbre couvrant minimum 2.2
    def heuristic_cover_min(self):
        graph = self.couvrant(self.graph)
        change = True
        while change:
            change = False
            tmp = []
            for node in graph.nodes:
                if node in self.free and graph.degree(node) == 1:
                    tmp.append(node)
            if len(tmp) > 0:
                for node in tmp:
                    graph.remove_node(node)
                change = True
        # self.draw_sousgraph(graph)
        return graph, self.weight(graph)

    # Randomisation des heuristiques de construction 2.3
    def graph_to_individu(self, graph):
        individu = []
        for node in self.free:
            if node in graph.nodes():
                individu.append(1)
            else:
                individu.append(0)
        return individu

    def random_graph(self):
        graph = self.graph.copy()
        for a, b in graph.edges:
            if random.randint(0, 1) == 1:
                graph[a][b]['weight'] += graph[a][b]['weight'] * random.uniform(0.05, 0.2)
            else:
                graph[a][b]['weight'] -= graph[a][b]['weight'] * random.uniform(0.05, 0.2)
        return graph

    def random_individu_from_PCC(self):
        graph, _ = SteinerGraphe(self.random_graph(), self.terminaux).heuristic_PCC()
        return self.graph_to_individu(graph)

    def random_individu_from_cover_min(self):
        graph, _ = SteinerGraphe(self.random_graph(), self.terminaux).heuristic_cover_min()
        return self.graph_to_individu(graph)

    def generate_randomisation(self):
        population = []
        for i in range(int(N / 2)):
            i1 = self.random_individu_from_PCC()
            i2 = self.random_individu_from_cover_min()
            if i1 not in population:
                population.append(i1)
            if i2 not in population:
                population.append(i2)
        self.population = self.population_sorted(population)
        if len(self.population) == 0:  # pour etre sur d'avoir une solution realisable
            self.population = self.population_sorted([[1 for i in range(len(self.free))]])
        return self.population[0]

    # 3. Recherche locale
    def recherche_locale(self):
        individu, score = self.generate()[0]
        change = True
        while change:
            change = False
            new_pop = []
            for i in range(len(self.free)):
                if individu[i] == 1:
                    if (self.graph.degree(self.free[i])) == 1:
                        new = list(individu)
                        new[i] = 0
                        new_pop.append(new)
                    else:
                        new = list(individu)
                        new[i] = 0
                        _, _, b = self.fitness(new)
                        if b:
                            new_pop.append(new)
                else:
                    if (self.graph.degree(self.free[i])) != 1:
                        new = list(individu)
                        new[i] = 1
                        _, _, b = self.fitness(new)
                        if b:
                            new_pop.append(new)
            if len(new_pop) > 0:
                pop = self.population_sorted(new_pop)
                if (pop[0][1] < score):
                    change = True
                    individu, score = pop[0]
        # print(individu, score)
        return individu, score


f = open('resultB', 'a')
for i in range(1, 19):
    name = ""
    if i < 10:
        name = "B/b0" + str(i) + ".stp"
    else:
        name = "B/b" + str(i) + ".stp"
    print(name)
    f.write(name + " :\n")
    g, t = readfile(name)
    sg = SteinerGraphe(g, t)
    v1, t1 = [], []
    for j in range(10):
        print(j)
        deb = time.clock()
        v = sg.algorithm_genetic(True, True)[1]
        fin = time.clock()
        v1.append(v)
        t1.append(fin - deb)
    for j in range(10):
        print(j)
        deb = time.clock()
        v = sg.algorithm_genetic(True, False)[1]
        fin = time.clock()
        v1.append(v)
        t1.append(fin - deb)
    for j in range(10):
        print(j)
        deb = time.clock()
        v = sg.algorithm_genetic(False, True)[1]
        fin = time.clock()
        v1.append(v)
        t1.append(fin - deb)
    for j in range(10):
        print(j)
        deb = time.clock()
        v = sg.algorithm_genetic(False, False)[1]
        fin = time.clock()
        v1.append(v)
        t1.append(fin - deb)
    f.write(
        "\tGEN\t" + str(np.mean(v1)) + "\t" + str(np.var(v1)) + "\t" + str(np.std(v1)) + "\t" + str(np.mean(t1)) + "\n")
    deb = time.clock()
    v = sg.heuristic_PCC()[1]
    fin = time.clock()
    f.write("\tPCC\t" + str(v) + "\t" + str(fin - deb) + "\n")
    deb = time.clock()
    v = sg.heuristic_cover_min()[1]
    fin = time.clock()
    f.write("\tCM\t" + str(v) + "\t" + str(fin - deb) + "\n")
    deb = time.clock()
    v = sg.generate_randomisation()[1]
    fin = time.clock()
    f.write("\tRAN\t" + str(v) + "\t" + str(fin - deb) + "\n")
    deb = time.clock()
    v = sg.recherche_locale()[1]
    fin = time.clock()
    f.write("\tLOC\t" + str(v) + "\t" + str(fin - deb) + "\n")
f.close()
"""
f = open('resultC', 'a')
for i in range(1, 21):
    name = ""
    if i < 10:
        name = "C/c0" + str(i) + ".stp"
    else:
        name = "C/c" + str(i) + ".stp"
    print(name)
    f.write(name + " :\n")
    g, t = readfile(name)
    sg = SteinerGraphe(g, t)
    v1, t1 = [], []
    for j in range(10):
        print(j)
        deb = time.clock()
        v = sg.algorithm_genetic(True, True)[1]
        fin = time.clock()
        v1.append(v)
        t1.append(fin - deb)
    for j in range(10):
        print(j)
        deb = time.clock()
        v = sg.algorithm_genetic(True, False)[1]
        fin = time.clock()
        v1.append(v)
        t1.append(fin - deb)
    for j in range(10):
        print(j)
        deb = time.clock()
        v = sg.algorithm_genetic(False, True)[1]
        fin = time.clock()
        v1.append(v)
        t1.append(fin - deb)
    for j in range(10):
        print(j)
        deb = time.clock()
        v = sg.algorithm_genetic(False, False)[1]
        fin = time.clock()
        v1.append(v)
        t1.append(fin - deb)
    f.write(
        "\tGEN\t" + str(np.mean(v1)) + "\t" + str(np.var(v1)) + "\t" + str(np.std(v1)) + "\t" + str(np.mean(t1)) + "\n")
    deb = time.clock()
    v = sg.heuristic_PCC()[1]
    fin = time.clock()
    f.write("\tPCC\t" + str(v) + "\t" + str(fin - deb) + "\n")
    deb = time.clock()
    v = sg.heuristic_cover_min()[1]
    fin = time.clock()
    f.write("\tCM\t" + str(v) + "\t" + str(fin - deb) + "\n")
    deb = time.clock()
    v = sg.generate_randomisation()[1]
    fin = time.clock()
    f.write("\tRAN\t" + str(v) + "\t" + str(fin - deb) + "\n")
    deb = time.clock()
    v = sg.recherche_locale()[1]
    fin = time.clock()
    f.write("\tLOC\t" + str(v) + "\t" + str(fin - deb) + "\n")
f.close()

f = open('resultD', 'a')
for i in range(1, 21):
    name = ""
    if i < 10:
        name = "D/d0" + str(i) + ".stp"
    else:
        name = "D/d" + str(i) + ".stp"
    print(name)
    f.write(name + " :\n")
    g, t = readfile(name)
    sg = SteinerGraphe(g, t)
    v1, t1 = [], []
    for j in range(10):
        print(j)
        deb = time.clock()
        v = sg.algorithm_genetic(True, True)[1]
        fin = time.clock()
        v1.append(v)
        t1.append(fin - deb)
    for j in range(10):
        print(j)
        deb = time.clock()
        v = sg.algorithm_genetic(True, False)[1]
        fin = time.clock()
        v1.append(v)
        t1.append(fin - deb)
    for j in range(10):
        print(j)
        deb = time.clock()
        v = sg.algorithm_genetic(False, True)[1]
        fin = time.clock()
        v1.append(v)
        t1.append(fin - deb)
    for j in range(10):
        print(j)
        deb = time.clock()
        v = sg.algorithm_genetic(False, False)[1]
        fin = time.clock()
        v1.append(v)
        t1.append(fin - deb)
    f.write(
        "\tGEN\t" + str(np.mean(v1)) + "\t" + str(np.var(v1)) + "\t" + str(np.std(v1)) + "\t" + str(np.mean(t1)) + "\n")
    deb = time.clock()
    v = sg.heuristic_PCC()[1]
    fin = time.clock()
    f.write("\tPCC\t" + str(v) + "\t" + str(fin - deb) + "\n")
    deb = time.clock()
    v = sg.heuristic_cover_min()[1]
    fin = time.clock()
    f.write("\tCM\t" + str(v) + "\t" + str(fin - deb) + "\n")
    deb = time.clock()
    v = sg.generate_randomisation()[1]
    fin = time.clock()
    f.write("\tRAN\t" + str(v) + "\t" + str(fin - deb) + "\n")
    deb = time.clock()
    v = sg.recherche_locale()[1]
    fin = time.clock()
    f.write("\tLOC\t" + str(v) + "\t" + str(fin - deb) + "\n")
f.close()

f = open('resultE', 'a')
for i in range(1, 21):
    name = ""
    if i < 10:
        name = "E/e0" + str(i) + ".stp"
    else:
        name = "E/e" + str(i) + ".stp"
    print(name)
    f.write(name + " :\n")
    g, t = readfile(name)
    sg = SteinerGraphe(g, t)
    v1, t1 = [], []
    for j in range(10):
        print(j)
        deb = time.clock()
        v = sg.algorithm_genetic(True, True)[1]
        fin = time.clock()
        v1.append(v)
        t1.append(fin - deb)
    for j in range(10):
        print(j)
        deb = time.clock()
        v = sg.algorithm_genetic(True, False)[1]
        fin = time.clock()
        v1.append(v)
        t1.append(fin - deb)
    for j in range(10):
        print(j)
        deb = time.clock()
        v = sg.algorithm_genetic(False, True)[1]
        fin = time.clock()
        v1.append(v)
        t1.append(fin - deb)
    for j in range(10):
        print(j)
        deb = time.clock()
        v = sg.algorithm_genetic(False, False)[1]
        fin = time.clock()
        v1.append(v)
        t1.append(fin - deb)
    f.write(
        "\tGEN\t" + str(np.mean(v1)) + "\t" + str(np.var(v1)) + "\t" + str(np.std(v1)) + "\t" + str(np.mean(t1)) + "\n")
    deb = time.clock()
    v = sg.heuristic_PCC()[1]
    fin = time.clock()
    f.write("\tPCC\t" + str(v) + "\t" + str(fin - deb) + "\n")
    deb = time.clock()
    v = sg.heuristic_cover_min()[1]
    fin = time.clock()
    f.write("\tCM\t" + str(v) + "\t" + str(fin - deb) + "\n")
    deb = time.clock()
    v = sg.generate_randomisation()[1]
    fin = time.clock()
    f.write("\tRAN\t" + str(v) + "\t" + str(fin - deb) + "\n")
    deb = time.clock()
    v = sg.recherche_locale()[1]
    fin = time.clock()
    f.write("\tLOC\t" + str(v) + "\t" + str(fin - deb) + "\n")
f.close()
"""
