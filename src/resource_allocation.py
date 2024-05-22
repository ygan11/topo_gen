import json
import random
import statistics
from utils import *
from config import abs_file_path , avg_qubit_per_repeater
from topo_resilience_eval import convert_networkx_graph_to_string_link_failure, convert_networkx_graph_to_string_node_failure

def allocate_resource(topo_file_name, k, topo_allocated_file_name, graph_type=0):
    if graph_type == 0:
        readDirName = abs_file_path + '/dist/topos/'
    elif graph_type == 1:
        readDirName = abs_file_path + '/dist/topos/link_failure/'
    else:
        readDirName = abs_file_path + '/dist/topos/map_size/'
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
    # assert total_resource_units >= 0
    print(f"total_resource_units before resource allocator: {total_resource_units}")
    print(f"crt total repater qubits num:", sum([G_repeater.nodes[node]['num_qubits'] for node in G_repeater.nodes()]))
    # resource_allocator(G_repeater, k_shortest_paths, k, total_resource_units)
    resource_allocator_betweenness_centrality(G_repeater, k_shortest_paths, k, total_resource_units)

    total_repeater_qubits = sum([G_repeater.nodes[node]['num_qubits'] for node in G_repeater.nodes()])
    print(f"total_resource_number: {total_resource_number}, total_repeater_qubits: {total_repeater_qubits}, ")
    assert abs(sum([G_repeater.nodes[node]['num_qubits'] for node in G_repeater.nodes()]) - total_resource_number) <= 2

    # graph_plot(G)

    # Save the graph to a file
    # saveDirPath = '../dist/topos_resourced/'
    # convert_networkx_graph_to_string(G=G, G_repeater=G_repeater, filename=(saveDirPath + topo_allocated_file_name))
    if type != 1:  # 0: normal, 1: link failure, 2: map size
        convert_networkx_graph_to_string(G=G, G_repeater=G_repeater, filename=(topo_allocated_file_name), graph_type=graph_type)
    else:# elif graph_type == 1:
        link_failure_rates = [0.01, 0.02, 0.04, 0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.20, 0.25]#[0.25]#[0.12, 0.15, 0.18, 0.20, 0.25]#[0.01, 0.02, 0.04, 0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.20, 0.25]#, 0.12, 0.15, 0.2]

        for link_failure_rate in link_failure_rates:
            convert_networkx_graph_to_string_link_failure(G=G, G_repeater=G_repeater, filename=(topo_allocated_file_name), failure_rate=link_failure_rate)

    # for node_failure_rate in node_failure_rate:
    #     convert_networkx_graph_to_string_node_failure(G=G, G_repeater=G_repeater, filename=(topo_allocated_file_name), node_failure_rate=node_failure_rate)


def allocate_resource_greedy(topo_file_name, k, topo_allocated_file_name, graph_type=0):
    if graph_type == 0:
        readDirName = abs_file_path + '/dist/topos/'
    elif graph_type == 1:
        readDirName = abs_file_path + '/dist/topos/link_failure/'
    else:
        readDirName = abs_file_path + '/dist/topos/map_size/'
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
    # assert total_resource_units >= 0
    print(f"total_resource_units before resource allocator: {total_resource_units}")
    print(f"crt total repater qubits num:", sum([G_repeater.nodes[node]['num_qubits'] for node in G_repeater.nodes()]))
    # resource_allocator(G_repeater, k_shortest_paths, k, total_resource_units)
    resource_allocator_betweenness_centrality(G_repeater, k_shortest_paths, k, total_resource_units)

    total_repeater_qubits = sum([G_repeater.nodes[node]['num_qubits'] for node in G_repeater.nodes()])
    print(f"total_resource_number: {len(G_repeater.nodes()) * avg_qubit_per_repeater}, total_repeater_qubits: {total_repeater_qubits}, ")
    assert abs(sum([G_repeater.nodes[node]['num_qubits'] for node in G_repeater.nodes()]) - len(G_repeater.nodes()) * avg_qubit_per_repeater) <= 2

    # graph_plot(G)


    convert_networkx_graph_to_string(G=G, G_repeater=G_repeater, filename=(topo_allocated_file_name), graph_type=graph_type)




def allocate_resource_uniformly(topo_file_name, topo_allocated_file_name, graph_type=0):
    if graph_type == 0:
        readDirName = abs_file_path + '/dist/topos/'
    elif graph_type == 1:
        readDirName = abs_file_path + '/dist/topos/link_failure/'
    else:
        readDirName = abs_file_path + '/dist/topos/map_size/'
    # readDirName = abs_file_path + '/dist/topos/'
    with open(readDirName + topo_file_name, 'r') as file:
        graph_data = json.load(file)
    G = nx.node_link_graph(graph_data)

    G_repeater = repeaters_graph(G)


    total_resource_units = len(G_repeater.nodes()) * avg_qubit_per_repeater #total_resource_number#init_rnode_resource_info(G_complete=G, G_repeater=G_repeater)

    resource_allocator_uniformly(G_repeater, total_resource_units)

    # graph_plot(G)

    if type != 1:  # 0: normal, 1: link failure, 2: map size
        if topo_file_name.startswith("deepPlace"):
            print("uniform")
            convert_networkx_graph_to_string(G=G, G_repeater=G_repeater, filename=(topo_allocated_file_name), graph_type=graph_type, uniform=True)
        else:
            convert_networkx_graph_to_string(G=G, G_repeater=G_repeater, filename=(topo_allocated_file_name), graph_type=graph_type)
    else:
        link_failure_rates = [0.01, 0.02, 0.04, 0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.20, 0.25]#[0.25]#[0.12, 0.15, 0.18, 0.20, 0.25]#[0.01, 0.02, 0.04, 0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.20, 0.25]#, 0.12, 0.15, 0.2]

        for link_failure_rate in link_failure_rates:
            convert_networkx_graph_to_string_link_failure(G=G, G_repeater=G_repeater, filename=(topo_allocated_file_name), failure_rate=link_failure_rate)



def calculate_node_count(k, k_shortest_paths):
    # Initialize the node_count dictionary
    node_count = {}

    for i in range(0, k):
        # Iterate through each pair of source and target nodes
        for source, target in k_shortest_paths:
            # Check if the i-th shortest path exists
            if i < len(k_shortest_paths[(source, target)]):
                # Iterate through each node in the i-th shortest path
                for n in k_shortest_paths[(source, target)][i]:
                    # Initialize the dictionary for this node if it doesn't exist
                    if n not in node_count:
                        node_count[n] = {}
                    # Initialize the count for this path index if it doesn't exist
                    if str(i) not in node_count[n]:
                        node_count[n][str(i)] = 0
                    # Increment the count
                    node_count[n][str(i)] += 1

    return node_count


def init_rnode_resource_info(G_complete, G_repeater):#, node_count, k_shortest_paths, k):

    

    for node in G_repeater.nodes():
        # Get the number of qubits for neighboring endnodes
        neighbor_qubits = [G_complete.nodes[endnode]['num_qubits'] for endnode in G_complete.neighbors(node) if
                           G_complete.nodes[endnode]['type'] == 'endnode']

        # Check if the list is empty
        if neighbor_qubits:
            # Calculate min, max, and median if there are neighboring endnodes
            G_repeater.nodes[node]['min_width'] = 2#min(neighbor_qubits) #* 2
            G_repeater.nodes[node]['max_width'] = max(neighbor_qubits) #* 2
            G_repeater.nodes[node]['median'] = statistics.median(neighbor_qubits) #* 2
        else:
            # assert False
            # print(f"No neighboring endnodes for node: {node}")
            # Handle the case where there are no neighboring endnodes
            # You can set default values or handle it as per your requirements
            default_min_value = 2
            default_max_value = 10
            default_median_value = 7
            G_repeater.nodes[node]['min_width'] = default_min_value  # Replace with your default value
            G_repeater.nodes[node]['max_width'] = default_max_value  # Replace with your default value
            G_repeater.nodes[node]['median'] = default_median_value  # Replace with your default value

    # add a new attr to each repeater node in G_repeater, which is the demanding score of the repeater node
    # the demanding score is determined by the number of endnodes that are connected to the repeater node
    # and the number of shortest paths that pass through the repeater node
    total_resource_num = len(G_repeater.nodes()) * avg_qubit_per_repeater#total_resource_number
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
    alpha_1 = 0.7
    alpha_2 = 0.2
    alpha_3 = 0.1
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


def all_k_betweenness_centrality(G, k_shortest_paths, k):
    # Initialize the betweenness_centrality dictionary
    all_k_betweenness_centrality = {}

    # Calculate the node count
    node_count = calculate_node_count(k, k_shortest_paths)

    for ki in range(k):
        # Initialize the betweenness_centrality dictionary for the i-th shortest path
        k_betweenness_centrality = {}

        for n in G.nodes():
            if n not in k_betweenness_centrality:
                k_betweenness_centrality[n] = 0
            k_betweenness_centrality[n] = node_count[n][str(ki)]

        # Normalize the betweenness centrality
        max_bc = max(k_betweenness_centrality.values())
        min_bc = min(k_betweenness_centrality.values())

        # Normalize the betweenness centrality
        # bc_sum = sum(k_betweenness_centrality.values())
        # # for node in G.nodes():
        # #     k_betweenness_centrality[node] = k_betweenness_centrality[node] / bc_sum

        for n in G.nodes():
            k_betweenness_centrality[n] = (k_betweenness_centrality[n] - min_bc) / (max_bc - min_bc)

        all_k_betweenness_centrality[ki] = k_betweenness_centrality



    return all_k_betweenness_centrality

def resource_allocator_uniformly(G_repeater, total_resource_num):
    # allocate resource to each repeater node in G_repeater
    # the resource is evenly allocated to each repeater node

    single_resource_num = int(total_resource_num / len(G_repeater.nodes()))
    # print(f"single_resource_num: {single_resource_num}")

    for node in G_repeater.nodes():
        # Randomly assign the resource to each repeater node in range [10, 15]
        G_repeater.nodes[node]['num_qubits'] = random.randint(10, 15) # single_resource_num
        
def resource_allocator_betweenness_centrality(G_repeater, k_shortest_paths, k, total_resource_num):
    # allocate resource to each repeater node in G_repeater
    # the resource is determined by the betweenness centrality of the node
    alpha_k = [0.7, 0.2, 0.1]
    all_k_bc = all_k_betweenness_centrality(G_repeater, k_shortest_paths, k)
    # Calculate the betweenness centrality of each node
    nodes_score = {}
    for rn in G_repeater.nodes():
        for ki in range(0, k): 
            if rn not in nodes_score:
                nodes_score[rn] = 0.0
            nodes_score[rn] += alpha_k[ki] * all_k_bc[ki][rn]

    print("n=0:", all_k_bc[0][100], all_k_bc[1][100], all_k_bc[2][100])
    # k = 0
    print("nodes_score:", nodes_score)         

    # for node in G_repeater.nodes():

    score_sum = sum(nodes_score.values())
    assigned_qubits_num = 0
    for n in G_repeater.nodes(): 
        # print(int(total_resource_num * (nodes_score[n] / score_sum) ))
        # total_num += int(total_resource_num * (nodes_score[n] / score_sum) )
        G_repeater.nodes[n]['num_qubits'] += int(total_resource_num * (nodes_score[n] / score_sum) )
        assigned_qubits_num += int(total_resource_num * (nodes_score[n] / score_sum) )

    resource_num = total_resource_num - assigned_qubits_num
    print("total resource_num:", total_resource_num, "repeater_qubits:", sum([G_repeater.nodes[node]['num_qubits'] for node in G_repeater.nodes()]))
    # Sort the nodes based on the score
    repeater_nodes = sorted(G_repeater.nodes(), key=lambda x: nodes_score[x], reverse=True)
    while resource_num > 0:
        for node in repeater_nodes:
            G_repeater.nodes[node]['num_qubits'] += 1
            resource_num -= 1
            if resource_num == 0:
                break

    # allocate resource to each repeater node in G_repeater
    # total_num = 0


    # print("total_num:", total_num)