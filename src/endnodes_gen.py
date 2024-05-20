from config import map_size
import networkx as nx
import random
import json
from config import abs_file_path
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


# A function to create a 2D map with size map_size * map_size
def endnode_graph_gen_different_map_size(n, topoIdx, map_size):

    seed = 8848 + topoIdx
    random.seed(seed)
    G = nx.Graph()
    # Randomly generate n nodes with 2D coordinates, and add them to the graph
    for i in range(n):
        #num_qubits = random.randint(1, 10)
        G.add_node(i, pos=(map_size * random.random(), map_size * random.random()), type='endnode', num_qubits=random.randint(10, 30))

    dirPath = abs_file_path + '/dist/endnodes/map_size/'
    fileName = 'endnodesLocs-' + str(map_size) + '-' + str(n) + '-' + str(topoIdx) + '.json'
    with open(dirPath + fileName, 'w') as file:
        json.dump(nx.node_link_data(G), file)

    file.close()

    return dirPath + fileName
