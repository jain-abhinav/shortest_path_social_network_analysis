import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import time

#general purpose function to catch integer error
def int_error_testing(any_input):
    try:
        any_input = int(any_input)
    except:
        any_input = "invalid"
    return any_input

# Reading G as an adjacency list from file
def prepare_adj_list(graph_choice):
    adj_list = {}
    with open("{}_graph.txt".format(graph_choice)) as f:
        for line in f:
            if line.startswith('#'):
                next(f)
            else:
                if graph_choice == "florentine_family":
                    from_node, to_node, distance = line.split()
                    distance = int(distance)
#                    print(from_node, to_node, distance)
                else:
                    from_node, to_node, distance = map(int, line.split())
                adj_list.setdefault(from_node, {})[to_node] = distance
    return adj_list

# Djikstra's Shortest Path Algorithm
class Heap(object):
    def __init__(self):
        self.minheap = []

    def __str__(self):
        return str(self.minheap)

    def __len__(self):
	    return len(self.minheap)

    def is_member(self, v):
	    return v in self.minheap

    def get_index(self, v):
	    return self.minheap.index(v)

    def insert_node(self, node, D):
        self.minheap.append(node)
        last = len(self.minheap) - 1
        self.siftup(last, D)
        return self.minheap

    def delmin(self, D):
        min = self.minheap.pop(0)
        if len(self.minheap) > 1:
            self.minheap.insert(0, self.minheap[-1])
            del self.minheap[-1]
            self.siftdown(0, D)
        return min

    def siftup(self, child, D):
        while child > 0:
            parent = child // 2
            if D[self.minheap[parent]] > D[self.minheap[child]]:
                self.minheap[parent], self.minheap[child] = self.minheap[child], self.minheap[parent]
                child = parent
            else:
                break
        return self.minheap

    def siftdown(self, parent, D):
        child = 2 * parent + 1
        if child < (len(self.minheap) - 1) and D[self.minheap[child + 1]] < D[self.minheap[child]]:
            child += 1
        while child <= (len(self.minheap) - 1):
            if D[self.minheap[parent]] > D[self.minheap[child]]:
                self.minheap[parent], self.minheap[child] = self.minheap[child], self.minheap[parent]
                parent = child
                child = 2 * parent + 1
            else:
                break
        return self.minheap

def djikstra_list(G, root_node): # Djikstra's Shortest Path - List Implementation
    S = list(G.keys()) # List of nodes, unordered with respect to their shortest distances from root_node
    S.remove(root_node) # Initial list of temporary nodes
    D = {root_node:0} # Dict of shortest distances from root_node, initializing with permanent label for root_node
    p = {root_node:None} # Dict of parents, initializing with permanent label for root_node
	#print(G[root_node])
    for u in S:
        R = G[root_node] # Getting dict of neighbors and their distances from the root_node
        D[u] = R.get(u, float('inf')) # Adding temporary labels for temporary nodes - distances for neighbors and infinity for rest
        p[u] = root_node # Adding temporary labels for temporary nodes - making root_node parent of all
	#print(D)
	#print(p)
    while len(S) != 0:
        S_with_D = {i:D[i] for i in D if i in S} # Creating dictionary with nodes in S as keys and their D values as key values
        u = min(S_with_D, key = S_with_D.get) # Node in S that has minimum D value
        S.remove(u) # Removing it from temporary list and making it permanent
        for v in G[u].keys(): # for every vertex in adjacency list of u

            if D[v] > D[u] + G[u][v]:
                D[v] = D[u] + G[u][v] # Relaxing inequality
                p[v] = u
    return D,p

def djikstra_heap(G, root_node): # Djikstra's Shortest Path - Heap Implementation
	non_root_nodes = list(G.keys())
	non_root_nodes.remove(root_node) # Creating dict of all non-root nodes
	D = {root_node:0} # Dict of shortest distances from root_node, initializing with permanent label for root_node
	p = {root_node:None} # Dict of parents, initializing with permanent label for root_node
	for u in non_root_nodes:
		D[u] = float('inf') # Adding temporary labels for non-root nodes - infinity for all
		p[u] = root_node # Adding temporary labels for temporary nodes - making root_node parent of all

	S = Heap() # Creating empty heap
	S.insert_node(root_node, D) # Initially adding root-node only

	while len(S) != 0:
		u = S.delmin(D) # top heap node with minimum D value
		for v in G[u].keys(): # for every vertex in adjacency list of u
			if D[v] > D[u] + G[u][v]:
				D[v] = D[u] + G[u][v] # Relaxing inequality
				p[v] = u
				if S.is_member(v) == False: # if v not in S
					S.insert_node(v, D)
				else:
					v_index = S.get_index(v)
					S.siftup(v_index, D)
	print("\nD = ", D)
	print("\np = ", p)

def get_root_node():
    while True:
        root_node = "None"
        choice = input("Enter Root Node Value (Between 0 and 199):")
        if choice == "":
            break
        root_node = int_error_testing(choice)
        if root_node == "invalid" or root_node < 0 or root_node > 199:
            print("\nInappropriate Input.")
        else:
            break
    return root_node

def implementation_choice(adj_list, graph_choice):
    while True:
        print("\n{} implementation would be better for {} graph.".format("Array" if graph_choice == "Dense" else "Heap", graph_choice))
        choice = input("1. Choose Array Implementation:\n2. Choose Heap Implementation:\nInput Here:")
        if choice == "":
            break
        choice = int_error_testing(choice)
        if choice == "invalid":
            print("\nInappropriate Input.")
        elif choice == 1:
            root_node = get_root_node()
            if root_node != "None":
                D = {}
                p = {}
                D, p = djikstra_list(adj_list, root_node)
                print("\nD = ", D)
                print("\np = ", p)
        elif choice == 2:
            root_node = get_root_node()
            if root_node != "None":
                djikstra_heap(adj_list, root_node)
        else:
            print("\nInappropriate Input.")

def graph_choose(graph_choice = "None"):
    while True:
        choice = input("\n1. Choose Sparse Graph:\n2. Choose Dense Graph:\nInput Here:")
        if choice == "":
            break
        choice = int_error_testing(choice)
        if choice == "invalid":
            print("\nInappropriate Input.")
        elif choice == 1:
            graph_choice = "Sparse"
            adj_list = prepare_adj_list(graph_choice)
            implementation_choice(adj_list, graph_choice)
        elif choice == 2:
            graph_choice = "Dense"
            adj_list = prepare_adj_list(graph_choice)
            implementation_choice(adj_list, graph_choice)
        else:
            print("\nInappropriate Input.")

# Generating list of arcs representation of a graph and saving it to text file
def create_graph(order_n, initial_arcs, graph_type, flag):
    G = nx.DiGraph(nx.barabasi_albert_graph(order_n, initial_arcs, seed = 200)) # Generating directed graph
    graph_size = len(G.edges)
    weights = list(np.random.randint(1, 5, size = graph_size))   # Generating list of random weights for graph
    weights = [(weight, ) for weight in weights]    # Converting every element in list weights to a tuple
    arcs_and_weights = [("#", "#", "{}".format(flag)), ("#", "#", "n = {}".format(order_n))]    # inserting flag for Stream and number of arcs
    arcs_and_weights += ([list(G.edges)[i] + weights[i] for i in range(graph_size)])    # Concatenating each tuple element in arcs with corresponding tuple element in weights
    print("{} Nodes and {} Arcs in {} Graph.".format(order_n, graph_size, graph_type))
    file = open('{}_graph.txt'.format(graph_type), 'w')
    for item in arcs_and_weights:
    	file.write("%s %s %s\n" % item)
    file.close()

def graph_parameters():
    order_n = 200   # Number of nodes
    SG_initial_arcs = 2  # Number of arcs to attach from a new node to existing nodes for a sparse graph
    DG_initial_arcs = 180 # Number of arcs to attach from a new node to existing nodes for a dense graph
    print("\n")
    #Sparse graph
    create_graph(order_n, SG_initial_arcs, "Sparse", "S")
    #Dense graph
    create_graph(order_n, DG_initial_arcs, "Dense", "D")

def dijktra():
    graph_parameters()
    graph_choose()

# Ballyskate Street Graph - BSG
def roller_blade():
    BSG_adj_list = {1: {2: 6, 3: 10, 4: 11}, 2: {1: 6, 4: 3, 6: 6, 7: 12}, 3: {1: 10, 4: 5, 5: 8, 9: 9}, 4: {1: 11, 2: 3, 3: 5, 5: 7, 6: 2, 9: 12}, 5: {3: 8, 4: 7, 6: 4, 8: 2, 9: 3}, 6: {2: 6, 4: 2, 5: 4, 7: 7, 8: 9}, 7: {2: 12, 6: 7, 8: 4, 'p': 11}, 8: {5: 2, 6: 9, 7: 4, 'p': 7}, 9: {3: 9, 4: 12, 5: 3, 'p': 10}, 'p': {7: 11, 8: 7, 9: 10}}
    while True:
        print("\nHeap implementation would be better for Sparse graph.")
        choice = input("1. Choose Array Implementation:\n2. Choose Heap Implementation:\nInput Here:")
        if choice == "":
            break
        choice = int_error_testing(choice)
        if choice == "invalid":
            print("\nInappropriate Input.")
        elif choice == 1:
            D = {}
            p = {}
            D, p = djikstra_list(BSG_adj_list, 1)
            print("\nD = ", D)
            print("\np = ", p)
        elif choice == 2:
            djikstra_heap(BSG_adj_list, 1)
        else:
            print("\nInappropriate Input.")

def mag_get_d_p(MAG):
    edge_size = len(MAG.edges)
    node_size = MAG.order()
    weights = [(1, ) for weight in [1]*edge_size]    # Generating list of tuples with weight 1 for each adge
    arcs_and_weights = []#("#", "#", "n = {}".format(node_size))]    # inserting flag for Stream and number of arcs
    arcs_and_weights += ([list(MAG.edges)[i] + weights[i] for i in range(edge_size)])    # Concatenating each tuple element in arcs with corresponding tuple element in weights
    print("\n{} Nodes and {} Arcs in Graph.".format(node_size, edge_size))
    file = open('florentine_family_graph.txt', 'w')
    for item in arcs_and_weights:
    	file.write("%s %s %s\n" % item)
    file.close()
    adj_list = prepare_adj_list("florentine_family")
    D = {}
    p = {}
    counter = 0
    for nodes in MAG.nodes(data = False):
        #for nodes part of a component
        if nodes in adj_list.keys():
            D[nodes], p[nodes] = djikstra_list(adj_list, nodes)
        #for nodes which are not connected
        else:
            counter += 1
            D_dict = {nodes: 0}
            D_dict.update({name:float('inf') for name in MAG.nodes(data = False) if name != nodes})
            D[nodes] = D_dict
            p[nodes] = {name:None for name in MAG.nodes(data = False)}
    return D,p, counter, node_size

def closeness_central(D, counter, node_size):
    table_close = pd.DataFrame.from_dict(D)
    closeness = {}
    for columns in table_close.columns:
        closeness[columns] = (node_size - counter - 1) / np.nansum(table_close[columns])
    return closeness

def florentine_family():
    # Using networkx to create florentine family graph
    MAG = nx.florentine_families_graph()
    MAG.add_node("Pucci")
    MAG = nx.DiGraph(MAG)
    D, p, counter, node_size = mag_get_d_p(MAG) #Getting the D and p values using Djikstra's list algorithm
    closeness = closeness_central(D, counter, node_size)    #cacluating closeness centrality
    print("\nNormalized Closeness centrality:", closeness)
    print("\nNormalized Betweenness centrality:", nx.betweenness_centrality(MAG, normalized = True))  #NEED TO BE DONE
    print("\nNormalized Degree Centrality:", nx.degree_centrality(MAG))
    print("\nNormalized Eigen-vector Centrality:", nx.eigenvector_centrality(MAG))
    nx.draw(MAG, with_labels=True)
    plt.show()

def main():
    while True:
        choice = input("\n1. Dijktra's Shortest Path Algorithm:\n2. Roller-blading Race in Ballyskat Problem:\
        \n3. Florentine Families Marriage Alliances Problem:\nInput Here:")
        if choice == "":
            break
        choice = int_error_testing(choice)
        if choice == "invalid":
            print("\nInappropriate Input.")
        elif choice == 1:
            dijktra()
        elif choice == 2:
            roller_blade()
        elif choice == 3:
            florentine_family()
        else:
            print("\nInappropriate Input.")

if __name__ == "__main__":
    main()
