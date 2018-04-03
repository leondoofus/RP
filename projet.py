import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
V = ['v1','v2','v3','v4','v5','u1','u2','u3','u4']
T = ['v1','v2','v3','v4','v5']
E = [
	('v1','v2',{'weight':8}),
	('v1','v3',{'weight':9}),
	('v1','u1',{'weight':2}),
	('v1','u3',{'weight':2}),
	('v2','v5',{'weight':5}),
	('v2','u1',{'weight':2}),
	('v2','u4',{'weight':8}),
	('v3','v4',{'weight':8}),
	('v3','u2',{'weight':4}),
	('v3','u3',{'weight':8}),
	('v4','u2',{'weight':3}),
	('v5','u2',{'weight':5}),
	('v5','u4',{'weight':8}),
	('u1','u2',{'weight':1})]
G.add_nodes_from(V)
G.add_edges_from(E)

def kruskal(g,v,t,chaine):
	l = [x for x in V if x not in T]
	print (l)
	new = nx.Graph()
	weight = 0
	for a, b, data in sorted(g.edges(data=True)):
		#print (a,b,data['weight'])
		if not ((a in l and chaine[l.index(a)] == 0) or (b in l and chaine[l.index(b)] == 0)):
			new.add_edge(a,b,weight=data['weight'])
			weight += data['weight']
			if nx.algorithms.cycles.cycle_basis(new) :
				new.remove_edge(a,b)
				weight -= data['weight']
	print (weight)
	print (nx.algorithms.components.is_connected(new)) #connexe
	nx.draw_networkx(new, with_labels=True)
	plt.show()
	return (new, weight, nx.algorithms.components.is_connected(new))

#nx.draw_networkx(G, with_labels=True)
#plt.show()
kruskal(G,V,T,[1,1,0,0])