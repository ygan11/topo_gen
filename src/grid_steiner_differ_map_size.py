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
from utils import convert_networkx_graph_to_string, extract_endnode_file_name_differ_map_size
#from utils import *



def construct_steiner_tree_different_map_size(endnodes_graph_file, map_size=1000, grid_size=15, l_rr=200):

    G, _ = read_endnodes_init_grid_graph_without_edges(endnodes_graph_file=endnodes_graph_file, map_size=map_size, grid_size=grid_size)
    endnodes = [node for node in G.nodes if G.nodes[node]['type'] == 'endnode']
    repeaters = [node for node in G.nodes if G.nodes[node]['type'] == 'repeater']

    assert len(endnodes) + len(repeaters) == len(G.nodes)

    # Add edges to G for every pair of nodes
    # if the distance between two repeaters is less than l_rr
    for node1 in repeaters:
        for node2 in repeaters:
            if node1 != node2:
                dis = math.sqrt((G.nodes[node1]['pos'][0] - G.nodes[node2]['pos'][0]) ** 2 + (G.nodes[node1]['pos'][1] - G.nodes[node2]['pos'][1]) ** 2)
                if dis < 220:#l_rr:
                    G.add_edge(node1, node2, dis=dis, type='repeater')

    for endnode in endnodes:
        for repeater in repeaters:
            dis = math.sqrt((G.nodes[endnode]['pos'][0] - G.nodes[repeater]['pos'][0]) ** 2 + (G.nodes[endnode]['pos'][1] - G.nodes[repeater]['pos'][1]) ** 2)
            if dis < 220:#l_er:
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
    dirPath = abs_file_path + '/dist/topos/map_size/'
    map_size, endnode_num, topo_idx = extract_endnode_file_name_differ_map_size(endnodes_graph_file) #extract_endnode_file_name(endnodes_graph_file)
    with open(dirPath + "steiner-" + str(map_size) + '-' + str(endnode_num) + '-' + str(topo_idx) + '.json', 'w') as file:
        json.dump(nx.node_link_data(G), file)


def read_endnodes_init_grid_graph_without_edges(endnodes_graph_file, map_size=1000, grid_size=15):
    # Remove all edges
    # grid.remove_edges_from(G.edges())
    # Add all edges if the distance between two nodes is less than l_rr
    # Add all edges if the distance between two nodes is less than l_rr
    grid = nx.grid_2d_graph(grid_size, grid_size)

    # for node1 in grid.nodes:
    #     for node2 in grid.nodes:
    #         if node1 != node2:
    #             if math.sqrt((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2) < l_rr:
    #                 grid.add_edge(node1, node2)

    # Calculate the intersection points' 2-D position in a map_sz x map_sz map
    step_size = map_size / grid_size
    intersection_points = []
    for node in grid.nodes:
        x = (node[0] + 0.5) * step_size
        y = (node[1] + 0.5) * step_size
        intersection_points.append((x, y))

    #print(intersection_points)

    # Add nodes to the graph
    for node, pos in zip(grid.nodes, intersection_points):
        grid.nodes[node]['pos'] = pos
        grid.nodes[node]['type'] = 'repeater'

    G = nx.Graph()

    with open(endnodes_graph_file, 'r') as f:
        endnodes_graph = json.load(f)
        nodes = endnodes_graph['nodes']

    for node in nodes:
        if node['type'] == 'endnode':
            pos = node['pos']
            num_qubits = node['num_qubits']
            G.add_node(node['id'], pos=pos, num_qubits=num_qubits, type='endnode')
    endnodes = [node for node in G.nodes if G.nodes[node]['type'] == 'endnode']

    # Add repeaters to G
    id_r = len(endnodes)
    for node, pos in zip(grid.nodes, intersection_points):
        G.add_node(id_r, pos=pos, type='repeater')
        id_r += 1

    # Connect endnodes to repeaters if the distance is less than l_er
    repeaters = [node for node in G.nodes if G.nodes[node]['type'] == 'repeater']

    # Return the graph and the endnodes
    return G, endnodes

