
from utils import map_size
import networkx as nx
import random
import json
# A function to create a 2D map with size map_size * map_size
def endnode_graph_gen(n, topoIdx):

    seed = 8848 + topoIdx
    random.seed(seed)
    G = nx.Graph()
    # Randomly generate n nodes with 2D coordinates, and add them to the graph
    for i in range(n):
        #num_qubits = random.randint(1, 10)
        G.add_node(i, pos=(map_size * random.random(), map_size * random.random()), type='endnode', num_qubits=random.randint(10, 30))

    dirPath = './dist/endnodes/'
    fileName = 'endnodesLocs-' + str(n) + '-' + str(topoIdx) + '.json'
    with open(dirPath + fileName, 'w') as file:
        json.dump(nx.node_link_data(G), file)

    file.close()

    return dirPath + fileName
