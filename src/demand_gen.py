
import random
import networkx as nx
import math

from config import max_single_dis, c_fiber
from utils import yen_k_shortest_paths, read_endnodes_init_grid_graph_without_edges, add_dis_attr_to_edges


class Demand:  
    # A class to generate demands and test if they are satisfied
    def __init__(self, endnodes_graph_file, num_demands):
        self.demands_list = self.demand_gen(endnodes_graph_file=endnodes_graph_file, num_demands=num_demands)
     
    # a function recive a networkX graph and generate a list of demands like "n1, n2, fidelity\belongs to [0.85,1]"
    def demand_gen(self, endnodes_graph_file, num_demands):
        demands = []
        G, endnodes = read_endnodes_init_grid_graph_without_edges(endnodes_graph_file)
        #endnodes = [node for node in G.nodes if G.nodes[node]['type'] == 'endnode']

        for i in range(num_demands):
            source = random.choice(endnodes)
            target = random.choice(endnodes)
            while source == target:
                target = random.choice(endnodes)

            # make sure the source is < target
            if source > target:
                source, target = target, source

            s_pos = G.nodes[source]['pos']
            t_pos = G.nodes[target]['pos']
            line_dis = ((s_pos[0] - t_pos[0]) ** 2 + (s_pos[1] - t_pos[1]) ** 2) ** 0.5
            min_hop_num = line_dis // max_single_dis

            fidelity = math.pow(0.99, min_hop_num) / 1.2#random.uniform(0.85, 1)
            rate = 0.0#(c_fiber / max_single_dis) * ((0.5) ** (min_hop_num * 2)) * (0.5 ** (min_hop_num * 2 + 1)) / 100.0#random.uniform(0.85, 1)
            demands.append((source, target, fidelity, rate))
        # self.demands_list = demands
        return demands

    def demand_satisfying_test(self, G, demands):
        all_satisfied = True
        G = add_dis_attr_to_edges(G)
        for demand in demands:
            source, target, fidelity, rate = demand
            k = 3
            k_shortest_paths = yen_k_shortest_paths(G, source, target, k)
            for kp in k_shortest_paths:
                all_satisfied = self.is_satisfied(G, kp, demand) and all_satisfied
            
        return all_satisfied


    def r_calculate(self, G, path):
        # R = \frac { c _ { \text {fiber } } } { L } \left( \frac { 1 } { 2 } \right) ^ { N } \left[ 1 - \left( 1 - \frac { 1 } { 2 } e ^ { - L / L _ { \mathrm { att } } } \right) ^ { M } \right] ^ { N + 1 }
        # c_fiber = 200000.0
        max_single_hop_dis = max([G.edges[path[i], path[i+1]]['dis'] for i in range(len(path)-1)])
        N = len(path) - 2 # number of repeaters
        r = (c_fiber / max_single_hop_dis) * ((0.5) ** N) * (0.5 ** (N+1))
        return r

    def f_calculate(self, G, path):
        # F = \frac { 1 } { 4 } \left[ 1 + 3 \left( \frac { 4 F _ { \text {link } } - 1 } { 3 } \right) ^ { N + 1 } \right]
        F_link = 0.99
        N = len(path) - 2 # number of repeaters
        F = 0.25 * (1 + 3 * ((4 * F_link - 1) / 3) ** (N+1))
        return F

    def is_satisfied(self, G, path, demand):
        source, target, fidelity, rate = demand
        r_succeed = self.r_calculate(G, path) >= rate
        f_succeed = self.f_calculate(G, path) >= fidelity
        if not r_succeed:
            print("Rate not satisfied, rate is ", self.r_calculate(G, path), "demand rate is ", rate)
        if not f_succeed:
            print("Fidelity not satisfied", self.f_calculate(G, path), "demand fidelity is ", fidelity)
        if r_succeed and f_succeed:
            return True

        return False