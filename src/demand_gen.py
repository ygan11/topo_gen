
import random
import networkx as nx
import math

from config import max_single_dis

# a function recive a networkX graph and generate a list of demands like "n1, n2, fidelity\belongs to [0.85,1]"
def demand_gen(G, num_demands):
    demands = []
    endnodes = [node for node in G.nodes if G.nodes[node]['type'] == 'endnode']

    for i in range(num_demands):
        source = random.choice(endnodes)
        target = random.choice(endnodes)
        while source == target:
            target = random.choice(endnodes)
        s_pos = G.nodes[source]['pos']
        t_pos = G.nodes[target]['pos']
        line_dis = ((s_pos[0] - t_pos[0]) ** 2 + (s_pos[1] - t_pos[1]) ** 2) ** 0.5
        min_hop_num = line_dis // max_single_dis

        fidelity = math.pow(0.99, min_hop_num) / 1.05#random.uniform(0.85, 1)
        rate = random.uniform(0.85, 1)
        demands.append((source, target, fidelity, rate))
    return demands
