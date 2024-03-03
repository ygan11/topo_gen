import numpy as np
import networkx as nx
import random
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from itertools import combinations
from math import sqrt, exp
import json

from utils import extract_endnode_file_name, graph_plot

def dyn_search(x_min, x_max, y_target, f, f_is_increasing, precision):
    x = (x_min + x_max) / 2
    step = x
    
    for _ in range(100):
        step /= 2
        y = f(x)
        if abs(y - y_target) < abs(precision):
            break
        
        if ((y > y_target) != f_is_increasing):
            x += step
        else:
            x -= step
    return x

def construct_waxman(endnodes_graph_file, degree, edge_len):
    node_locs = []
    node_id_locs_mapping = {}

    

    # Read the graph from the file
    dirPath = '../dist/topos/'

    # Split the endnodes_graph_file to get the number of endnodes and the index of the topo
    endnode_num, topo_idx = extract_endnode_file_name(endnodes_graph_file)

    with open(dirPath + "deepPlace-" + str(endnode_num) + '-' + str(topo_idx) + '.json', "r") as f:

        data = json.load(f)
        G_raw = nx.json_graph.node_link_graph(data)

    # # Get the repeater node locations
    # for node in G_raw.nodes():
    #     if G_raw.nodes[node]["type"] == "repeater":
    #         node_locs.append(G_raw.nodes[node]['pos'])
    #         #node_id_locs_mapping[G_raw.nodes[node]["id"]] = G_raw.nodes[node]['pos']

    # node_locs.sort(key=lambda x: x[0] + int(x[1] * 10 / edge_len) * 1000000)

    for node in G_raw.nodes():
        node_id_locs_mapping[node] = G_raw.nodes[node]['pos']

    # get the number of endnode in G_raw, whose type is endnode
    endnode_num = len([node for node in G_raw.nodes() if G_raw.nodes[node]["type"] == "endnode"])
    repeater_num = len([node for node in G_raw.nodes() if G_raw.nodes[node]["type"] == "repeater"])
    controlling_d = sqrt(edge_len ** 2 / repeater_num)



    # Copy G_raw to G'
    G = nx.Graph()
    # Copy the only the repeater nodes from G_raw to G
    for node in G_raw.nodes():
        if G_raw.nodes[node]["type"] == "repeater":
            # print(f"node: {node}, pos: {G_raw.nodes[node]['pos']}, type: {G_raw.nodes[node]['type']}")
            G.add_node(node, pos=G_raw.nodes[node]['pos'], type=G_raw.nodes[node]["type"])


    def f(beta):
        edge_count = 0
        for n1, n2 in combinations(range(endnode_num, endnode_num + repeater_num), 2):
            l1, l2 = node_id_locs_mapping[n1], node_id_locs_mapping[n2]
            d = euclidean(l1, l2)
            
            if d < 2 * controlling_d:
                l = min(random.random() for _ in range(50))
                r = exp(-beta * d)
                if l < r:
                    edge_count += 1
                    G.add_edge(n1, n2, dis=d, type="repeater")
                    #links.append((n1, n2))
        
        return 2 * edge_count / repeater_num
    
    beta = dyn_search(0.0, 20.0, degree, f, False, 0.2)
    
    
    graph_plot(G)

    # Ensuring connectivity
    for cc in sorted(nx.connected_components(G), key=len, reverse=True)[1:]:
        print(f"cc: {cc}")
        for to_connect in random.sample(cc, min(3, len(cc))):
            # nearest = min(G.nodes() - cc, key=lambda x: euclidean(node_locs[x], node_locs[to_connect]))
            nearest = min(G.nodes() - cc, key=lambda x: euclidean(node_id_locs_mapping[x], node_id_locs_mapping[to_connect]))
            G.add_edge(nearest, to_connect, dis=euclidean(node_id_locs_mapping[nearest], node_id_locs_mapping[to_connect]), type="repeater")
            print(f"Connecting {nearest} to {to_connect}")
    # print(f"Is G connected: {nx.is_connected(G)}")
    assert nx.is_connected(G)
    print(f"Average degree: {sum(G.degree(node) for node in G.nodes()) / len(G.nodes())}")
    # Ensuring minimum degree
    for node in G.nodes():
        if G.degree(node) < degree:
            # potential_nodes = sorted((n for n in G.nodes() if n != node),
            #                          key=lambda x: euclidean(node_locs[x], node_locs[node]))
            potential_nodes = sorted((n for n in G.nodes() if n != node),
                                     key=lambda x: euclidean(node_id_locs_mapping[x], node_id_locs_mapping[node]))
            for p_node in potential_nodes[:degree - G.degree(node)]:
                G.add_edge(node, p_node)



    # Remove all repeater edges in G_raw
    for edge in G_raw.edges():
        if G_raw.nodes[edge[0]]["type"] == "repeater" and G_raw.nodes[edge[1]]["type"] == "repeater":
            G_raw.remove_edge(edge[0], edge[1])

    # Add the repeater edges in G to G_raw
    for edge in G.edges():
        G_raw.add_edge(edge[0], edge[1], dis=euclidean(node_id_locs_mapping[edge[0]], node_id_locs_mapping[edge[1]]), type="repeater")

    G = G_raw

    # print(f"endnode_num: {endnode_num}")

    graph_plot(G)
    # assert all repeater nodes are connected to at least one other repeater node
    repeater_nodes = [node for node in G.nodes() if G.nodes[node]["type"] == "repeater"]
    for node in repeater_nodes:
        has_repeater_neighbor = False
        for neighbor in G.neighbors(node):
            if G.nodes[neighbor]["type"] == "repeater":
                has_repeater_neighbor = True
                break
        if not has_repeater_neighbor:
            assert False, f"Repeater node {node} is not connected to any other repeater node"
            # Add an edge between the repeater node and the nearest repeater node 
            # nearest_repeater = min(repeater_nodes - node, key=lambda x: euclidean(node_id_locs_mapping[x], node_id_locs_mapping[node]))

    # # Generating topo description
    # description = f"{n}\n{alpha}\n{q}\n{k}\n"
    # description += "\n".join(f"{random.randint(10, 14)} {' '.join(map(str, loc))}" for loc in node_locs)
    # description += "\n"
    # description += "\n".join(f"{u} {v} {random.randint(3, 7)}" for u, v in G.edges)
    # endnode_num, topo_idx = extract_endnode_file_name(endnodes_graph_file)
    with open(dirPath + "waxman-" + str(endnode_num) + '-' + str(topo_idx) + '.json', 'w') as f:
        data = nx.json_graph.node_link_data(G)
        json.dump(data, f)
