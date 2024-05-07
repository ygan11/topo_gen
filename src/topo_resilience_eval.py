import json
import random
import networkx as nx
import statistics
from utils import *


def G_removed_edges(topo_file_name, failure_rate):
    readDirName = '../dist/topos/'
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
    readDirName = '../dist/topos_resourced/'
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
    avg_path_length = nx.average_shortest_path_length(G_repeater, weight='dis')#statistics.mean(path_lengths)
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
    print('Average Path Length:', avg_path_length)
    print('Standard Deviation of Path Lengths:', std_dev_path_length)
    print('Maximum Path Length:', max_path_length)
    print('Minimum Path Length:', min_path_length)
    print('Average Degree:', avg_degree)
    print("=====================================")




    


    






    # Save the graph to a file
    # saveDirPath = '../dist/topos_resourced_failed/'
    # convert_networkx_graph_to_string(G=G, G_repeater=G_repeater, filename=(saveDirPath + topo_allocated_file_name))