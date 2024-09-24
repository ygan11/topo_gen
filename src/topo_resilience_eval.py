import json
import random
import networkx as nx
import statistics
from utils import *
from config import abs_file_path
import numpy as np
import random
# from resource_allocation import init_rnode_resource_info, resource_allocator

def G_removed_edges(topo_file_name, failure_rate):
    readDirName = abs_file_path + '/dist/topos/'
    with open(readDirName + topo_file_name, 'r') as file:
        graph_data = json.load(file)
    G = nx.node_link_graph(graph_data)

    G_failed = G.copy()
    
    # remove edges from the graph until the failure rate is reached
    edges = list(G.edges())
    edges_to_remove = int(failure_rate * len(edges))
    for i in range(edges_to_remove):
        edge = random.choice(edges)
        G_failed.remove_edge(edge[0], edge[1])
        edges.remove(edge)

    return G_failed


def G_removed_nodes(topo_file_name, failure_rate):

    readDirName = abs_file_path + '/dist/topos_resourced/'
    with open(readDirName + topo_file_name, 'r') as file:
        graph_data = json.load(file)
    G = nx.node_link_graph(graph_data)

    G_failed = G.copy()
    
    # remove nodes from the graph until the failure rate is reached
    nodes = list(G.nodes())
    nodes_to_remove = int(failure_rate * len(nodes))
    for i in range(nodes_to_remove):
        node = random.choice(nodes)
        G_failed.remove_node(node)
        nodes.remove(node)

    return G_failed


def get_path_statistics(G, k):
    G_repeater = repeaters_graph(G)
    
    # Find k-shortest paths between all pairs of nodes in the graph
    k_shortest_paths = {}
    for source in G_repeater.nodes():
        for target in G_repeater.nodes():
            if source != target:
                k_shortest_paths[(source, target)] = yen_k_shortest_paths(G_repeater, source, target, k)

    # Get the CDF of the 1-shortest path lengths
    path_lengths = []
    for source, target in k_shortest_paths:
        for path in k_shortest_paths[(source, target)]:
            path_lengths.append(len(path))
    path_lengths.sort()
    cdf = {}
    for i in range(len(path_lengths)):
        cdf[path_lengths[i]] = i / len(path_lengths)

    # Get the average path length
    avg_path_length = nx.average_shortest_path_length(G_repeater)#statistics.mean(path_lengths)
    # Get the standard deviation of the path lengths
    std_dev_path_length = statistics.stdev(path_lengths)
    # Get the maximum path length
    max_path_length = max(path_lengths)
    # Get the minimum path length
    min_path_length = min(path_lengths)

    # Get the average degree of Graph
    avg_degree = sum(dict(G_repeater.degree()).values()) / len(G_repeater.nodes())

    # Plot the discrete CDF
    plt.plot(list(cdf.keys()), list(cdf.values()))
    plt.xlabel('Path Length')
    plt.ylabel('CDF')
    plt.title('CDF of 1-Shortest Path Lengths')
    plt.show()

    # Print the statistics
    print('Average Hop Count:', avg_path_length)
    print('Standard Deviation of Path Lengths:', std_dev_path_length)
    print('Maximum Hop Number:', max_path_length)
    print('Minimum Hop Number:', min_path_length)
    print('Average Degree:', avg_degree)
    print("=====================================")
    edges_dis = [G_repeater.edges[edge]['dis'] for edge in G_repeater.edges()]
    print("Average Edge dis:", np.mean(edges_dis))
    print("Max Edge dis:", max(edges_dis))
    print("Min Edge dis:", min(edges_dis))
    # print("Average Edge dis:", np.mean([G_repeater.edges[edge]['dis'] for edge in G_repeater.edges()]))

    print('repeater node number:', len(G_repeater.nodes()))
    print("=====================================")


def convert_networkx_graph_to_string_link_failure(G, G_repeater, filename, failure_rate):
    random.seed(8848)
    # Assign the attributes "num_qubits" of G by the attributes "num_qubits" of G_repeater if the node id is the same
    for node in G_repeater.nodes():
        G.nodes[node]["num_qubits"] = G_repeater.nodes[node]["num_qubits"]

    # folder_path = "../../quantum_topo_design_2024/QuantumRouting/topo_data/"
    folder_path = '/home/ygan11/quantum_topo_design_2024/QuantumRouting/topo_data/link_failure/'
    # replace the file extension with .txt
    failure_rate_name_mapping = {0.01: '1', 0.02: '2', 0.04: '4', 0.05: '5', 0.08: '8', 0.1: '10', 0.12: '12', 0.15: '15', 0.18: "18", 0.2: '20', 0.25: '25', 0.3: '30', 0.35: '35', 0.4: '40', 0.45: '45', 0.5: '50'}
    filename = filename.split('.')[0] + '-fr-'+ failure_rate_name_mapping[failure_rate] + '.txt'
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

            if random.random() < failure_rate:
                f.write(f'{n1} {n2} 0\n')
            else:
                f.write(f'{n1} {n2} 100\n')

            # for edge in G_repeater.edges():
        #     if G_repeater.edges[edge]["type"] == "repeater":
        #       #f.write(f'{edge[0]} {edge[1]} 999\n')
        #       f.write(f'{node_to_index[edge[0]]} {node_to_index[edge[1]]} 999\n')

    print(f'Link failure Graph saved to {folder_path + filename}')
    # close file
    f.close()


# def allocate_resource_link_failure(topo_file_name, k, topo_allocated_file_name, failure_rate):
#     readDirName = abs_file_path + '/dist/topos/'
#     with open(readDirName + topo_file_name, 'r') as file:
#         graph_data = json.load(file)
#     G = nx.node_link_graph(graph_data)

#     G_repeater = repeaters_graph(G)

#     # Find k-shortest paths between all pairs of nodes in the graph
#     # k = 2
#     k_shortest_paths = {}
#     for source in G_repeater.nodes():
#         for target in G_repeater.nodes():
#             if source != target:
#                 k_shortest_paths[(source, target)] = yen_k_shortest_paths(G_repeater, source, target, k)

#     total_resource_units = init_rnode_resource_info(G_complete=G, G_repeater=G_repeater)

#     resource_allocator(G_repeater, k_shortest_paths, k, total_resource_units)

#     # graph_plot(G)

#     # Save the graph to a file
#     # saveDirPath = '../dist/topos_resourced/'
#     # convert_networkx_graph_to_string(G=G, G_repeater=G_repeater, filename=(saveDirPath + topo_allocated_file_name))
#     convert_networkx_graph_to_string_link_failure(G=G, G_repeater=G_repeater, filename=(topo_allocated_file_name), failure_rate=failure_rate)


    


    






    # Save the graph to a file
    # saveDirPath = '../dist/topos_resourced_failed/'
    # convert_networkx_graph_to_string(G=G, G_repeater=G_repeater, filename=(saveDirPath + topo_allocated_file_name))




def convert_networkx_graph_to_string_node_failure(G, G_repeater, filename, failure_rate):
    random.seed(8848)
    # Assign the attributes "num_qubits" of G by the attributes "num_qubits" of G_repeater if the node id is the same
    for node in G_repeater.nodes():
        G.nodes[node]["num_qubits"] = G_repeater.nodes[node]["num_qubits"]

    # folder_path = "../../quantum_topo_design_2024/QuantumRouting/topo_data/"
    folder_path = '/home/ygan11/quantum_topo_design_2024/QuantumRouting/topo_data/'
    # replace the file extension with .txt
    failure_rate_name_mapping = {0.01: '1', 0.02: '2', 0.04: '4', 0.05: '5', 0.08: '8', 0.1: '10', 0.12: '12', 0.15: '15', 0.2: '20'}
    filename = filename.split('.')[0] + '-fr-'+ failure_rate_name_mapping[failure_rate] + '.txt'
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
                if random.random() > failure_rate:
                    f.write(
                        f'{G_repeater.nodes[node]["num_qubits"]} {G_repeater.nodes[node]["pos"][0]} {G_repeater.nodes[node]["pos"][1]} 1\n')
                else:
                    f.write(
                        f'1 {G_repeater.nodes[node]["pos"][0]} {G_repeater.nodes[node]["pos"][1]} 1\n')

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
