import networkx as nx
import matplotlib.pyplot as plt
import json
import math

from config import *
from itertools import islice


def convert_networkx_graph_to_string(G, G_repeater, filename):
    # Assign the attributes "num_qubits" of G by the attributes "num_qubits" of G_repeater if the node id is the same
    for node in G_repeater.nodes():
        G.nodes[node]["num_qubits"] = G_repeater.nodes[node]["num_qubits"]

    # folder_path = "../../quantum_topo_design_2024/QuantumRouting/topo_data/"
    folder_path = '/home/ygan11/quantum_topo_design_2024/QuantumRouting/topo_data/'
    # replace the file extension with .txt
    filename = filename.split('.')[0] + '.txt'
    with open(folder_path + filename, 'w') as f:
        f.write(f'{len(G.nodes())}\n')
        f.write(f'1.0\n')
        f.write(f'0.5\n')
        f.write(f'6\n')

        # counter = number of endnodes in G
        counter = len(G.nodes()) - len(G_repeater.nodes())
        for node in G.nodes():
            if G.nodes[node]["type"] == "endnode":
                assert "num_qubits" in G.nodes[node] 
                f.write(f'{G.nodes[node]["num_qubits"]} {G.nodes[node]["pos"][0]} {G.nodes[node]["pos"][1]} 0\n') # 0 means endnode

        for node in G_repeater.nodes():
            if G_repeater.nodes[node]["type"] == "repeater":
                assert "num_qubits" in G_repeater.nodes[node]
                f.write(
                    f'{G_repeater.nodes[node]["num_qubits"]} {G_repeater.nodes[node]["pos"][0]} {G_repeater.nodes[node]["pos"][1]} 1\n')

        # Create a mapping from node to index
        rnode_to_counter = {}
        # Create a mapping from repeater node id to counter
        for node in G_repeater.nodes():
            rnode_to_counter[node] = counter
            counter += 1

        for edge in G.edges():
            n1 = edge[0]
            n2 = edge[1]
            if G.nodes[n1]["type"] == "repeater":
                n1 = rnode_to_counter[n1]
            if G.nodes[n2]["type"] == "repeater":
                n2 = rnode_to_counter[n2]
            f.write(f'{n1} {n2} 100\n')

            # for edge in G_repeater.edges():
        #     if G_repeater.edges[edge]["type"] == "repeater":
        #       #f.write(f'{edge[0]} {edge[1]} 999\n')
        #       f.write(f'{node_to_index[edge[0]]} {node_to_index[edge[1]]} 999\n')

    print(f'Graph saved to {folder_path + filename}')
    # close file
    f.close()


def extract_endnode_file_name(filename):
    # filename format: endnodesLocs-numEndnodes-topoIdx.json
    # e.g. endnodesLocs-100-1.json

    # return numEndnodes, topoIdx
    return filename.split('-')[1], filename.split('-')[2].split('.')[0]


def graph_plot(G):
    # color_map = {'endnode': 'blue', 'repeater': 'gold'}
    color_map = {'endnode': '#10739E', 'repeater': '#B40504'}
    # draw the graph
    pos = nx.get_node_attributes(G, 'pos')
    node_colors = [color_map[G.nodes[n]['type']] for n in G.nodes()]
    edge_colors = [color_map[G.edges[e]['type']] for e in G.edges()]
    num_endnodes = len([n for n in G.nodes() if G.nodes[n]['type'] == 'endnode'])
    # only get labels for node which type is repeater

    #labels_qubits = {n: G.nodes[n]['num_qubits'] for n in G.nodes() if G.nodes[n]['type'] == 'repeater'}
    labels_id = {n: n for n in G.nodes()}
    # nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color=edge_colors, width=0.5, node_size=10, labels=labels_qubits)
    # nx.draw(G, pos, with_labels=False, node_color=node_colors, edge_color=edge_colors, width=0.5, node_size=10)
    label_endnodes = {n: n for n in G.nodes() if G.nodes[n]['type'] == 'endnode'}
    nx.draw(G, pos, with_labels=True, labels=label_endnodes, node_color=node_colors, edge_color=edge_colors, width=0.5, node_size=10)
    plt.show()
    # Pause the program until the plot is closed
    #plt.savefig('graph.png')



def yen_k_shortest_paths(G, source, target, k):
    return list(islice(nx.shortest_simple_paths(G, source, target, weight='dis'), k)) 
    # # Compute the shortest path using Dijkstra's algorithm
    # graph = G.copy()
    # # add 'dis' attribute to each edge, which is the distance between the two end nodes
    # for edge in graph.edges():
    #     graph[edge[0]][edge[1]]['dis'] = ((graph.nodes[edge[0]]['pos'][0] - graph.nodes[edge[1]]['pos'][0]) ** 2 + (
    #             graph.nodes[edge[0]]['pos'][1] - graph.nodes[edge[1]]['pos'][1]) ** 2) ** 0.5

    # shortest_path = nx.shortest_path(graph, source, target, weight='dis')
    # A = [shortest_path]
    # B = []

    # for i in range(1, k):
    #     for j in range(len(A[i - 1]) - 1):
    #         spur_node = A[i - 1][j]
    #         root_path = A[i - 1][:j + 1]

    #         # Remove edges from the graph that are part of previous shortest paths
    #         for path in A:
    #             if len(path) > j and root_path == path[:j + 1]:
    #                 graph.remove_edge(path[j], path[j + 1])

    #         # Calculate the spur path from the spur node to the target
    #         spur_path = nx.shortest_path(graph, spur_node, target, weight='dis')

    #         # Combine the root path and spur path to get a new candidate path
    #         candidate_path = root_path + spur_path[1:]

    #         # Add the candidate path to the list of potential k-shortest paths
    #         B.append(candidate_path)

    #         # Restore the removed edges to the graph
    #         for path in A:
    #             if len(path) > j and root_path == path[:j + 1]:
    #                 graph.add_edge(path[j], path[j + 1], dis=G[path[j]][path[j + 1]]['dis'])

    #     if len(B) == 0:
    #         # No more candidate paths to explore
    #         break

    #     # Sort the candidate paths by their total weight
    #     B.sort(key=lambda path: nx.path_weight(graph, path, weight='dis'), reverse=False)

    #     # Add the shortest candidate path to the list of k-shortest paths
    #     A.append(B[0])
    #     B.pop(0)

    # return A


def repeaters_graph(G):
    G_repeater = G.copy()
    # remove all endnodes and their edges from the graph G_repeater
    # remove all edges that are not between repeaters
    for edge in G.edges():
        if G.nodes[edge[0]]['type'] != 'repeater' or G.nodes[edge[1]]['type'] != 'repeater':
            G_repeater.remove_edge(edge[0], edge[1])

    for node in G.nodes():
        if G.nodes[node]['type'] == 'endnode':
            G_repeater.remove_node(node)
        else:
            G_repeater.nodes[node]['num_qubits'] = 0

    # calculate the distance between each pair of repeaters
    for edge in G_repeater.edges():
        node1 = G_repeater.nodes[edge[0]]
        node2 = G_repeater.nodes[edge[1]]
        G_repeater[edge[0]][edge[1]]['dis'] = ((node1['pos'][0] - node2['pos'][0]) ** 2 + (
                node1['pos'][1] - node2['pos'][1]) ** 2) ** 0.5

    for edge in G_repeater.edges():
        if 'dis' not in G_repeater.edges[edge]:
            print(f"Missing 'dis' attribute for edge: {edge}")

    # repeater_nodes = []
    # repeater_edges = []
    # for node in G.nodes():
    #     if G.nodes[node]['type'] == 'repeater':
    #         repeater_nodes.append(node)
    # for edge in G.edges():
    #     if G.nodes[edge[0]]['type'] == 'repeater' and G.nodes[edge[1]]['type'] == 'repeater':
    #         repeater_edges.append(edge)

    # G_repeater = nx.Graph()
    # G_repeater.add_nodes_from(repeater_nodes)
    # G_repeater.add_edges_from(repeater_edges)
    return G_repeater


def read_endnodes_init_grid_graph_without_edges(endnodes_graph_file):
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


def read_endnodes_init_grid_graph_with_grid_edges(endnodes_graph_file):
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

    grid_G_node_mappling = {}
    # Add repeaters to G
    id_r = len(endnodes)
    for node, pos in zip(grid.nodes, intersection_points):
        grid_G_node_mappling[node] = id_r
        G.add_node(id_r, pos=pos, type='repeater')
        id_r += 1

    # Add edges between repeaters in G to form a grid
    for edges in grid.edges:
        dis = ((grid.nodes[edges[0]]['pos'][0] - grid.nodes[edges[1]]['pos'][0]) ** 2 + (
                grid.nodes[edges[0]]['pos'][1] - grid.nodes[edges[1]]['pos'][1]) ** 2) ** 0.5
        G.add_edge(grid_G_node_mappling[edges[0]], grid_G_node_mappling[edges[1]], type='repeater', dis=dis)

    print(f'Number of nodes in G: {len(G.nodes)}')
    graph_plot(G)




    # Connect endnodes to repeaters if the distance is less than l_er
    repeaters = [node for node in G.nodes if G.nodes[node]['type'] == 'repeater']

    # Return the graph and the endnodes
    return G, endnodes


# def total_cost(G, print: bool = False):
def total_cost(G):
    total_cost = 0
    for edge in G.edges():
        total_cost += G[edge[0]][edge[1]]['dis'] * unit_fiber_cost

    for node in G.nodes():
        if G.nodes[node]['type'] == 'repeater':
            total_cost += unit_repeater_cost
    # if print:
    print(f'Total cost: {total_cost}')
    print(f'Number of repeaters: {len([node for node in G.nodes if G.nodes[node]["type"] == "repeater"])}')
    print(f'Total link length: {sum([G[edge[0]][edge[1]]["dis"] for edge in G.edges()])}')

    return total_cost


def add_dis_attr_to_edges(G):
    for edge in G.edges():
        G[edge[0]][edge[1]]['dis'] = ((G.nodes[edge[0]]['pos'][0] - G.nodes[edge[1]]['pos'][0]) ** 2 + (
                G.nodes[edge[0]]['pos'][1] - G.nodes[edge[1]]['pos'][1]) ** 2) ** 0.5
        
    return G



def graph_statistics(G):
    endnodes = [node for node in G.nodes if G.nodes[node]['type'] == 'endnode']
    print(f'Diameter: {nx.diameter(G)}')
    # only count the shortest path length between endnodes

    # print(f'Average shortest path length: {nx.average_shortest_path_length(G, weight='dis')}')
    print(f'Average hop count: {nx.average_shortest_path_length(G)}')
    print(f'Average degree: {sum([G.degree(node) for node in G.nodes]) / len(G.nodes)}')
    print(f'Average degree of repeaters: {sum([G.degree(node) for node in G.nodes if G.nodes[node]["type"] == "repeater"]) / len([node for node in G.nodes if G.nodes[node]["type"] == "repeater"])}')
    #print(f'Average degree of repeaters only connect to other repeaters: {sum([G.degree(node) for node in G.nodes if G.nodes[node]["type"] == "repeater" and all([G.nodes[neighbor]["type"] == "repeater" for neighbor in G.neighbors(node)])]) / len([node for node in G.nodes if G.nodes[node]["type"] == "repeater" and all([G.nodes[neighbor]["type"] == "repeater" for neighbor in G.neighbors(node)])])}')
    
    repeaters = [node for node in G.nodes if G.nodes[node]['type'] == 'repeater']
    repeater_edges = [edge for edge in G.edges if G.edges[edge]['type'] == 'repeater']
    print(f'Average degree of repeaters edges: {len(repeater_edges) / len(repeaters)}')
    # Get the max number of degree that a repeater node conncets to other repeater nodes
    max_degree = 0
    min_degree = 100
    # Print all node and its edges
    for repeater in repeaters:
        nbrs = G.neighbors(repeater)
        repeater_edges = 0
        for nbr in nbrs:
            if G.nodes[nbr]['type'] == 'repeater':
                repeater_edges += 1
        if repeater_edges > max_degree:
            max_degree = repeater_edges
        if repeater_edges < min_degree:
            min_degree = repeater_edges
    print(f'Maximum degree of repeater edges: {max_degree}')
    print(f'Minimum degree of repeater edges: {min_degree}')
    

