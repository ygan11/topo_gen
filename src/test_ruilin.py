import networkx as nx
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json
from collections import deque
from config import abs_file_path
map_size = 1000 # 1000
grid_size = 15  # 15
step_size = map_size / grid_size
color_map = {'end_node': '#10739E', 'repeater': '#B40504'}
def euclidean_distance(pos1, pos2):
    return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
def get_nearest_neighbors(graph, node, n_neighbors=2):
    node_pos = graph.nodes[node]['pos']
    distances = []

    for neighbor in graph.nodes:
        if neighbor != node:
            neighbor_pos = graph.nodes[neighbor]['pos']
            distance = euclidean_distance(node_pos, neighbor_pos)
            distances.append((neighbor, distance))

    distances.sort(key=lambda x: x[1])
    nearest_neighbors = [neighbor for neighbor, _ in distances[:n_neighbors]]
    return nearest_neighbors
def graph_plot(G):

    # draw the graph
    pos = nx.get_node_attributes(G, 'pos')
    node_colors = [color_map[G.nodes[n]['type']] for n in G.nodes()]
    edge_colors = [color_map[G.edges[e]['type']] for e in G.edges()]
    num_endnodes = len([n for n in G.nodes() if G.nodes[n]['type'] == 'end_node'])
    # only get labels for node which type is repeater

    # labels_qubits = {n: G.nodes[n]['num_qubits'] for n in G.nodes() if G.nodes[n]['type'] == 'repeater'}
    labels_id = {n: n for n in G.nodes()}
    # nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color=edge_colors, width=0.5, node_size=10, labels=labels_qubits)
    # nx.draw(G, pos, with_labels=False, node_color=node_colors, edge_color=edge_colors, width=0.5, node_size=10)
    label_endnodes = {n: n for n in G.nodes() if G.nodes[n]['type'] == 'end_node'}
    nx.draw(G, pos, with_labels=True, labels=label_endnodes, node_color=node_colors, edge_color=edge_colors,
            width=0.5, node_size=10)
    plt.show()
    # Pause the program until the plot is closed
    # plt.savefig('graph.png')
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

    # print(intersection_points)

    # Add nodes to the graph
    for node, pos in zip(grid.nodes, intersection_points):
        grid.nodes[node]['pos'] = pos
        grid.nodes[node]['xcoord'] = pos[0]
        grid.nodes[node]['ycoord'] = pos[1]
        grid.nodes[node]['type'] = 'repeater'

    G = nx.Graph()

    with open(endnodes_graph_file, 'r') as f:
        endnodes_graph = json.load(f)
        nodes = endnodes_graph['nodes']

    for node in nodes:
        if node['type'] == 'endnode':
            pos = node['pos']

            num_qubits = node['num_qubits']
            G.add_node(node['id'], pos=pos, num_qubits=num_qubits, type='end_node', xcoord = pos[0], ycoord =pos[1])
    endnodes = [node for node in G.nodes if G.nodes[node]['type'] == 'end_node']

    grid_G_node_mappling = {}
    # Add repeaters to G
    id_r = len(endnodes)
    for node, pos in zip(grid.nodes, intersection_points):
        grid_G_node_mappling[node] = id_r
        G.add_node(id_r, pos=pos, type='repeater', xcoord = pos[0], ycoord =pos[1])
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

class QuantumRepeaterDeployment:
    def __init__(self, nx_graph, Lmax, leaf_nodes=None):
        self.nx_graph = nx_graph
        self.Lmax = Lmax
        self.leaf_nodes = leaf_nodes if leaf_nodes else [n for n in nx_graph.nodes if nx_graph.degree[n] == 1]

    def choose_centers(self):
        Vleaf = self.leaf_nodes + [n for n in self.nx_graph.nodes if self.nx_graph.degree[n] == 1]
        # Vaccess = [n for n in self.nx_graph.nodes if any(leaf in Vleaf for leaf in self.nx_graph.neighbors(n)) and self.nx_graph.nodes[n]['type'] == 'repeater']
        Vaccess = set()
        repeater_nodes =[n for n in self.nx_graph.nodes if self.nx_graph.nodes[n]['type'] == 'repeater' ]
        for n in repeater_nodes:
            for leaf in self.nx_graph.neighbors(n):
                if leaf in Vleaf:
                    Vaccess.add(n)
        C = set()
        Vcovered = set()

        for v in Vaccess:
            # vleaf = [n for n in self.nx_graph.neighbors(v) if n in Vleaf]
            # if any(self.get_distance(l1, l2) > self.Lmax for l1 in vleaf for l2 in vleaf):
            #     C.add(v)
            #     Vcovered.update(self.get_coverage(v))
            neighbors = list(self.nx_graph.neighbors(v))
            vleaf = [n for n in neighbors if n in Vleaf]
            exceeds_Lmax = False
            for l1 in vleaf:
                for l2 in vleaf:
                    distance = self.get_distance(l1, l2)
                    if distance > self.Lmax:
                        exceeds_Lmax = True
                        break
                if exceeds_Lmax:
                    break
            if exceeds_Lmax:
                C.add(v)
                Vcovered.update(self.get_coverage_new(v))

        Vremaining = set(self.nx_graph.nodes) - Vcovered

        while Vremaining:
            max_coverage_node = None
            max_coverage_size = -1
            for node in Vremaining:
                coverage_size = len(self.get_coverage_new(node) & Vremaining)
                if coverage_size > max_coverage_size:
                    if self.nx_graph.nodes[node]['type'] == 'end_node':
                        continue
                    max_coverage_node = node
                    max_coverage_size = coverage_size

            v = max_coverage_node
            C.add(v)
            Vcovered.update(self.get_coverage_new(v))
            Vremaining = set(self.nx_graph.nodes) - Vcovered

        return C

    def get_distance(self, node1, node2):

        return nx.shortest_path_length(self.nx_graph, source=node1, target=node2, weight='dis')

    def get_coverage_old(self, node):

        return {n for n in self.nx_graph.nodes if self.get_distance(node, n) <= self.Lmax}

    def get_coverage_new(self, node):
        # Get all nodes within Lmax distance from the given node using BFS
        visited = {node}
        queue = deque([(node, 0)])
        coverage = set()

        while queue:
            current_node, current_distance = queue.popleft()
            if current_distance <= self.Lmax:
                coverage.add(current_node)
                for neighbor in self.nx_graph.neighbors(current_node):
                    if neighbor not in visited:
                        edge_distance = self.nx_graph[current_node][neighbor].get('dis', 1)
                        if current_distance + edge_distance <= self.Lmax:
                            visited.add(neighbor)
                            queue.append((neighbor, current_distance + edge_distance))

        return coverage

    def find_intermediate_nodes(self, centers):
        # MST = self.minimum_spanning_tree_with_intermediates_new(centers)
        # I = set()
        #
        # for edge in MST:
        #     nodes = self.get_nodes_on_edge(edge)
        #     node1 = nodes[0]
        #     for i in range(1, len(nodes)):
        #         node2 = nodes[i]
        #         if self.get_distance(node1, node2) > self.Lmax:
        #             I.add(nodes[i-1])
        #             node1 = nodes[i-1]
        # subgraph = self.nx_graph.copy()
        # nodes_to_avoid = [n for n in self.nx_graph.nodes if self.nx_graph.nodes[n]['type'] == 'end_node']
        # subgraph.remove_nodes_from(self.nodes_to_avoid)
        mst = nx.minimum_spanning_tree(self.nx_graph, weight='dis')
        I = set()
        for c1 in centers:
            for c2 in centers:
                edges_to_include = []
                if c1 != c2:
                    path = nx.shortest_path(mst, source=c1, target=c2)
                    for i in range(len(path) - 1):
                        edges_to_include.append((path[i], path[i + 1]))
                if edges_to_include:
                    nodes = [edge[0] for edge in edges_to_include] + [edges_to_include[-1][1]]
                    node1 = nodes[0]
                    # node1 = edges_to_include[0][0]
                    # for i in range(1, len(edges_to_include)):
                    #     node2 = edges_to_include[i][1]
                    #     if self.get_distance(node1, node2) > self.Lmax:
                    #         if node1 not in centers:
                    #             I.add(node1)
                    #         node1 = node2
                    for i in range(1, len(nodes)):
                        node2 = nodes[i]
                        if self.get_distance(node1, node2) > self.Lmax:
                            if nodes[i-1] not in centers and self.nx_graph.nodes[nodes[i-1]]['type'] == 'repeater':
                                I.add(nodes[i-1])
                            node1 = nodes[i-1]


        return I

    def minimum_spanning_tree(self, centers):
        subgraph = self.nx_graph.subgraph(centers)
        mst = nx.minimum_spanning_tree(subgraph)
        return list(mst.edges())

    def minimum_spanning_tree_with_intermediates(self, centers):
        complete_subgraph = self.nx_graph.subgraph(centers).copy()
        for center in centers:
            for node in self.nx_graph.nodes:
                if node not in centers:
                    complete_subgraph.add_edge(center, node, weight=self.get_distance(center, node))

        mst = nx.minimum_spanning_tree(complete_subgraph, weight='dis')
        return list(mst.edges())

    def minimum_spanning_tree_with_intermediates_new(self, centers):
        # Find the MST of the entire graph
        mst = nx.minimum_spanning_tree(self.nx_graph)

        # Extract the relevant edges and intermediate nodes to connect centers
        edges_to_include = []
        for c1 in centers:
            for c2 in centers:
                if c1 != c2:
                    path = nx.shortest_path(mst, source=c1, target=c2)
                    for i in range(len(path) - 1):
                        edges_to_include.append((path[i], path[i + 1]))

        return edges_to_include

    def get_nodes_on_edge(self, edge):
        # Return the nodes that form the edge
        return list(edge)

if __name__ == "__main__":
    file_path = abs_file_path + "/dist/endnodes/endnodesLocs-25-0.json"
    g, endnodes = read_endnodes_init_grid_graph_with_grid_edges(file_path)
    nearest_neighbors = {endnode: get_nearest_neighbors(g, endnode) for endnode in endnodes}
    for endnode in endnodes:
        for neighbor in nearest_neighbors[endnode]:
            # get distance of an edges
            dis = ((g.nodes[endnode]['pos'][0] - g.nodes[neighbor]['pos'][0]) ** 2 + (
                    g.nodes[endnode]['pos'][1] - g.nodes[neighbor]['pos'][1]) ** 2) ** 0.5
            g.add_edge(endnode, neighbor, type='end_node',dis = dis)
    print(endnodes)
    graph_plot(g)
    plt.show()
    Lmax = 100
    deployment = QuantumRepeaterDeployment(g, Lmax, endnodes)
    centers = deployment.choose_centers()
    inter_nodes = deployment.find_intermediate_nodes(centers)
    print(len(centers),centers)
    print(len(inter_nodes),inter_nodes)
#     get intersection of two sets
    print(len(inter_nodes-(centers & inter_nodes)))
#   plot center in original graph
    pos = nx.get_node_attributes(g, 'pos')
    node_colors = [color_map[g.nodes[n]['type']] for n in g.nodes()]
    edge_colors = [color_map[g.edges[e]['type']] for e in g.edges()]
    num_endnodes = len([n for n in g.nodes() if g.nodes[n]['type'] == 'end_node'])
    labels_id = {n: n for n in g.nodes()}
    label_endnodes = {n: n for n in g.nodes() if g.nodes[n]['type'] == 'end_node'}
    nx.draw(g, pos, with_labels=True, labels=label_endnodes, node_color=node_colors, edge_color=edge_colors,
            width=0.5, node_size=10)
    nx.draw_networkx_nodes(g, pos, nodelist=centers, node_color='r', node_size=50)
    plt.show()
    I = deployment.find_intermediate_nodes(centers)
    print(len(I))
    pos = nx.get_node_attributes(g, 'pos')
    node_colors = [color_map[g.nodes[n]['type']] for n in g.nodes()]
    edge_colors = [color_map[g.edges[e]['type']] for e in g.edges()]
    num_endnodes = len([n for n in g.nodes() if g.nodes[n]['type'] == 'end_node'])
    labels_id = {n: n for n in g.nodes()}
    label_endnodes = {n: n for n in g.nodes() if g.nodes[n]['type'] == 'end_node'}
    nx.draw(g, pos, with_labels=True, labels=label_endnodes, node_color=node_colors, edge_color=edge_colors,
            width=0.5, node_size=10)
    nx.draw_networkx_nodes(g, pos, nodelist=centers, node_color='r', node_size=50)
    nx.draw_networkx_nodes(g, pos, nodelist=I, node_color='g', node_size=50)
    plt.show()
#   plot inter_nodes in original graph
    pos = nx.get_node_attributes(g, 'pos')
    node_colors = [color_map[g.nodes[n]['type']] for n in g.nodes()]
    edge_colors = [color_map[g.edges[e]['type']] for e in g.edges()]
    num_endnodes = len([n for n in g.nodes() if g.nodes[n]['type'] == 'end_node'])
    labels_id = {n: n for n in g.nodes()}
    label_endnodes = {n: n for n in g.nodes() if g.nodes[n]['type'] == 'end_node'}
    nx.draw(g, pos, with_labels=True, labels=label_endnodes, node_color=node_colors, edge_color=edge_colors,
            width=0.5, node_size=10)
    nx.draw_networkx_nodes(g, pos, nodelist=I, node_color='g', node_size=50)
    plt.show()






