import json
import random
import statistics
from utils import *


def allocate_resource(topo_file_name, k, topo_allocated_file_name):
    readDirName = abs_file_path + '/dist/topos/'
    with open(readDirName + topo_file_name, 'r') as file:
        graph_data = json.load(file)
    G = nx.node_link_graph(graph_data)

    G_repeater = repeaters_graph(G)

    # Find k-shortest paths between all pairs of nodes in the graph
    # k = 2
    k_shortest_paths = {}
    for source in G_repeater.nodes():
        for target in G_repeater.nodes():
            if source != target:
                k_shortest_paths[(source, target)] = yen_k_shortest_paths(G_repeater, source, target, k)

    total_resource_units = init_rnode_resource_info(G_complete=G, G_repeater=G_repeater)

    resource_allocator(G_repeater, k_shortest_paths, k, total_resource_units)

    # graph_plot(G)

    # Save the graph to a file
    # saveDirPath = '../dist/topos_resourced/'
    # convert_networkx_graph_to_string(G=G, G_repeater=G_repeater, filename=(saveDirPath + topo_allocated_file_name))
    convert_networkx_graph_to_string(G=G, G_repeater=G_repeater, filename=(topo_allocated_file_name))


def calculate_node_count(k, k_shortest_paths):
    # Initialize the node_count dictionary
    node_count = {}

    for i in range(0, k):
        # Iterate through each pair of source and target nodes
        for source, target in k_shortest_paths:
            # Check if the i-th shortest path exists
            if i < len(k_shortest_paths[(source, target)]):
                # Iterate through each node in the i-th shortest path
                for node in k_shortest_paths[(source, target)][i]:
                    # Initialize the dictionary for this node if it doesn't exist
                    if node not in node_count:
                        node_count[node] = {}
                    # Initialize the count for this path index if it doesn't exist
                    if str(i) not in node_count[node]:
                        node_count[node][str(i)] = 0
                    # Increment the count
                    node_count[node][str(i)] += 1

    return node_count


def init_rnode_resource_info(G_complete, G_repeater):#, node_count, k_shortest_paths, k):

    total_resource_num = 1000

    for node in G_repeater.nodes():
        # Get the number of qubits for neighboring endnodes
        neighbor_qubits = [G_complete.nodes[endnode]['num_qubits'] for endnode in G_complete.neighbors(node) if
                           G_complete.nodes[endnode]['type'] == 'endnode']

        # Check if the list is empty
        if neighbor_qubits:
            # Calculate min, max, and median if there are neighboring endnodes
            G_repeater.nodes[node]['min_width'] = min(neighbor_qubits) * 2
            G_repeater.nodes[node]['max_width'] = max(neighbor_qubits) * 2
            G_repeater.nodes[node]['median'] = statistics.median(neighbor_qubits) * 2
        else:
            # assert False
            # print(f"No neighboring endnodes for node: {node}")
            # Handle the case where there are no neighboring endnodes
            # You can set default values or handle it as per your requirements
            default_min_value = 5
            default_max_value = 10
            default_median_value = 7
            G_repeater.nodes[node]['min_width'] = default_min_value  # Replace with your default value
            G_repeater.nodes[node]['max_width'] = default_max_value  # Replace with your default value
            G_repeater.nodes[node]['median'] = default_median_value  # Replace with your default value

    # add a new attr to each repeater node in G_repeater, which is the demanding score of the repeater node
    # the demanding score is determined by the number of endnodes that are connected to the repeater node
    # and the number of shortest paths that pass through the repeater node
    for node in G_repeater.nodes():
        G_repeater.nodes[node]['num_qubits'] = G_repeater.nodes[node]['min_width']
        total_resource_num -= G_repeater.nodes[node]['min_width']

    return total_resource_num

    # G_repeater's min and max width is determined by the endnodes that are connected to this repeater nodes
    # min_width = the smallest number of qubits that
    # for node in G_repeater.nodes():
    #   tt = [G_complete.nodes[endnode]['num_qubits'] for endnode in G_complete.neighbors(node) if G_complete.nodes[endnode]['type'] == 'endnode']
    #   G_repeater.nodes[node]['min_width'] =            min([G_complete.nodes[endnode]['num_qubits'] for endnode in G_complete.neighbors(node) if G_complete.nodes[endnode]['type'] == 'endnode']) * 2
    #   G_repeater.nodes[node]['max_width'] =            max([G_complete.nodes[endnode]['num_qubits'] for endnode in G_complete.neighbors(node) if G_complete.nodes[endnode]['type'] == 'endnode']) * 2
    #   G_repeater.nodes[node]['median'] = statistics.median([G_complete.nodes[endnode]['num_qubits'] for endnode in G_complete.neighbors(node) if G_complete.nodes[endnode]['type'] == 'endnode']) * 2


def resource_allocator(G_repeater, k_shortest_paths, k, total_resource_num):
    # allocate resource to each repeater node in G_repeater
    # the resource is determined by the number of endnodes that are connected to the repeater node
    # and the number of shortest paths that pass through the repeater node
    single_resource_num = 3  # int(total_resource_num / len(G_repeater.nodes()) / 5)
    # print(f"single_resource_num: {single_resource_num}")
    while total_resource_num > 0:

        # find the repeater node with the highest demanding score
        max_score = 0
        max_score_node = None
        for node in G_repeater.nodes():
            its_score = node_score(node=node, G_repeater=G_repeater, k_shortest_paths=k_shortest_paths, k=k,
                                   single_resource_num=single_resource_num)
            # print(f"node: {node}, its_score: {its_score}")
            if its_score > max_score:
                max_score = its_score
                max_score_node = node

        # allocate resource to the repeater node with the highest demanding score
        if max_score_node is None:
            print("max_score_node is None")
            max_score_node = random.choice(list(G_repeater.nodes()))
        # print(f"max_score_node: {max_score_node}, max_score: {max_score}")
        G_repeater.nodes[max_score_node]['num_qubits'] += single_resource_num
        total_resource_num -= single_resource_num


def node_score(node, G_repeater, k_shortest_paths, k, single_resource_num):
    # return the score of a node
    # the score is determined by the number of endnodes that are connected to the node
    # and the number of shortest paths that pass through the node
    alpha_1 = 3.0
    alpha_2 = 2.0
    alpha_3 = 1.0
    alphas = [alpha_1, alpha_2, alpha_3]

    score = 0.0
    crt_qubit = G_repeater.nodes[node]['num_qubits']
    # crt_satisfies_path_num = []
    next_satisfies_path_num = []
    for i in range(0, k):
        # count all the shortest paths that pass through this node and width <= crt_qubit
        satisfied_path_num = 0
        satisfiable_path_num = 0
        for source, target in k_shortest_paths:
            width = min(G_repeater.nodes[source]['num_qubits'], G_repeater.nodes[target]['num_qubits'])
            # Check if the i-th shortest path exists
            if i < len(k_shortest_paths[(source, target)]):
                # Iterate through each node in the i-th shortest path
                for node_ in k_shortest_paths[(source, target)][i]:
                    if node_ == node:
                        # if crt_qubit >= width:
                        #   satisfied_path_num += 1
                        if crt_qubit < width and crt_qubit + single_resource_num >= width:
                            satisfiable_path_num += 1
                        break
        # crt_satisfies_path_num.append(satisfied_path_num)
        next_satisfies_path_num.append(satisfiable_path_num)

    for i in range(0, k):
        score += alphas[i] * (next_satisfies_path_num[i])
        if (next_satisfies_path_num[i] == 0):
            score += (float(G_repeater.nodes[node]['num_qubits']) / float(G_repeater.nodes[node]['median'])) * alphas[i]

    return score
