import networkx as nx
import json
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.spatial import Delaunay

from sklearn.cluster import KMeans
from config import abs_file_path

import random
import multiprocessing
# import the function convert_to_txt() from the file utils.py
from utils import convert_networkx_graph_to_string
from utils import *

def construct_steiner_tree(endnodes_graph_file):

    G, _ = read_endnodes_init_grid_graph_without_edges(endnodes_graph_file=endnodes_graph_file)
    endnodes = [node for node in G.nodes if G.nodes[node]['type'] == 'endnode']
    repeaters = [node for node in G.nodes if G.nodes[node]['type'] == 'repeater']

    assert len(endnodes) + len(repeaters) == len(G.nodes)

    # Add edges to G for every pair of nodes
    # if the distance between two repeaters is less than l_rr
    for node1 in repeaters:
        for node2 in repeaters:
            if node1 != node2:
                dis = math.sqrt((G.nodes[node1]['pos'][0] - G.nodes[node2]['pos'][0]) ** 2 + (G.nodes[node1]['pos'][1] - G.nodes[node2]['pos'][1]) ** 2)
                if dis < 500:#l_rr:
                    G.add_edge(node1, node2, dis=dis, type='repeater')

    for endnode in endnodes:
        for repeater in repeaters:
            dis = math.sqrt((G.nodes[endnode]['pos'][0] - G.nodes[repeater]['pos'][0]) ** 2 + (G.nodes[endnode]['pos'][1] - G.nodes[repeater]['pos'][1]) ** 2)
            if dis < 500:#l_er:
                G.add_edge(endnode, repeater, dis=dis, type='endnode')

    # graph_plot(G)

    G_steiner_r1 = nx.algorithms.approximation.steiner_tree(G, terminal_nodes=endnodes, weight='dis', method="kou")
    # graph_plot(G_steiner_r1)
    # Get the repeater nodes used in the Steiner tree
    repeaters_steiner = [node for node in G_steiner_r1.nodes if G_steiner_r1.nodes[node]['type'] == 'repeater']
    G_steiner_r2 = nx.algorithms.approximation.steiner_tree(G, terminal_nodes=repeaters_steiner, weight='dis', method="kou")
    new_nodes = [node for node in G_steiner_r2.nodes if node not in repeaters_steiner]
    
    G_final = G_steiner_r1.copy()
    #G_final.add_nodes_from(new_nodes)
    for node in new_nodes:
        G_final.add_node(node, pos=G.nodes[node]['pos'], type='repeater')
    G_final.add_edges_from(G_steiner_r2.edges, type='repeater')

    G = G_final
    # 可视化
    repeater_node_num = len([node for node in G.nodes if G.nodes[node]['type'] == 'repeater'])
    # graph_plot(G)
    # print("Repeater node number: ", repeater_node_num)

    # Save the graph to a json file
    dirPath = abs_file_path + '/dist/topos/'
    endnode_num, topo_idx = extract_endnode_file_name(endnodes_graph_file)
    with open(dirPath + "steiner-" + str(endnode_num) + '-' + str(topo_idx) + '.json', 'w') as file:
        json.dump(nx.node_link_data(G), file)
