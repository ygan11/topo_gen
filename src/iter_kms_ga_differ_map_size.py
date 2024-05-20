import networkx as nx
import json
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.spatial import Delaunay, ConvexHull
from concave_hull import concave_hull, concave_hull_indexes

from sklearn.cluster import KMeans

from deap import base, creator, tools
#from scoop import futures
import random
import multiprocessing

from utils import read_endnodes_init_grid_graph_without_edges, l_er, l_rr, graph_plot #, extract_endnode_file_name, graph_plot, \
   # read_endnodes_init_grid_graph_with_grid_edges
from utils import graph_plot, graph_statistics, extract_endnode_file_name_differ_map_size
from demand_gen import Demand
from config import abs_file_path # map_size, step_size, 


class KmsGaDms:
    def __init__(self) -> None:
        pass
        # self.demands = demands
        # self.demands_list = demands.demands_list
        self.map_size = 1000
        self.grid_size = 15
        self.step_size = self.map_size / self.grid_size



    def iterate_kms_ga(self, endnodes_graph_file, map_size=1000, grid_size=15, l_rr=200):
        # print(type(map_size))
        # 读取endnodes_graph_file
        self.map_size = map_size
        self.grid_size = grid_size
        self.step_size = map_size / grid_size
        G, endnodes = read_endnodes_init_grid_graph_with_grid_edges_t(endnodes_graph_file=endnodes_graph_file, map_size=map_size, grid_size=grid_size)
        
        # print("The number of endnodes in the graph:", len(endnodes))
        ##-self.plot_and_metrics(G)
        # 获取初始图
        G = self.remove_repeaters_not_included_in_convex_hull(G, endnodes)
        # G = self.remove_repeaters_not_included_in_concave_hull(G, endnodes)
        ##-self.plot_and_metrics(G)

        G = self.copy_graph_with_reindex(G)
        P = self.get_start_graph(G)
        # 优化repeater位置
        self.optimize_repeater_pos(P, endnodes, l_er, l_rr)
        # 打印指标
        # plot_and_metrics(P)

        repeater_num = len([node for node in P.nodes if P.nodes[node]['type'] == 'repeater'])
        # no_illegal_edges = True
        min_repeaters_num = repeater_num
        num_clusters = repeater_num - 1

        min_P = P
        continues_failures_count = 0
        # while num_clusters > (repeater_num / 3):
        optimize_rate = [4, 2, 1]
        crt_rate = optimize_rate[0]
        while num_clusters > 0:
            # Perform clustering
            # clustering_repeater(num_clusters, P, repeater_positions)

            P = self.construct_new_graph(P, num_clusters)
            # plot_and_metrics(P)
            ##-print("num_clusters:", num_clusters)
            self.optimize_repeater_pos(P, endnodes, l_er, l_rr)
            ##-self.plot_and_metrics(P)
            ##-graph_statistics(P)
            if self.is_legal(P):
                continues_failures_count = 0
                min_P = P
            else:
                continues_failures_count += 1
                if continues_failures_count > (6 - crt_rate):#4:
                    if crt_rate == 1:
                        break
                    else:
                        continues_failures_count = 0
                        crt_rate = optimize_rate[optimize_rate.index(crt_rate) + 1]
                        P = min_P
                        self.optimize_repeater_pos_t(P, endnodes, l_er, l_rr)#min_P
                        num_clusters = len([node for node in P.nodes if P.nodes[node]['type'] == 'repeater']) - crt_rate
                        ##-print("re-caculate from num_clusters:", num_clusters)
                        continue
                    # break

            if num_clusters < min_repeaters_num:
                min_repeaters_num = num_clusters
            num_clusters -= crt_rate#1
            ##-print("current rate: ", crt_rate)

        self.optimize_repeater_pos_t(min_P, endnodes, l_er, l_rr)
        ##-print("final topology:")
        ##-self.plot_and_metrics(min_P)

        # Save this graph to a file
        # with open('./source/topologies/topo_candi/graph0' + '.json', 'w') as file:
        dirPath = abs_file_path + '/dist/topos/map_size/'
        _, endnode_num, topo_idx = extract_endnode_file_name_differ_map_size(endnodes_graph_file)#self.extract_endnodes_file_name_different_map_size(endnodes_graph_file)
        with open(dirPath + "deepPlace-" + str(map_size) + '-' + str(endnode_num) + '-' + str(topo_idx) + '.json', 'w') as file:
            json.dump(nx.node_link_data(min_P), file)

        return min_P


    # Get a start graph quickly for the optimization from initial grid graph
    def get_start_graph(self, G):


        # nx.draw(G, with_labels=False, node_size=10)
        node_positions = []
        for node in G.nodes:
            node_positions.append(list(G.nodes[node]['pos']))
        points = np.array(node_positions)
        tri = Delaunay(points)
        ##-print("number of nodes in the graph:", len(G.nodes))
        # remove all edges from G
        # print("beofre remove edges:", G.edges())
        # plot_and_metrics(G)
        # G.remove_edges_from(G.edges())
        # Add edges to G
        for triangle in tri.simplices:
            for i in range(3):
                u = triangle[i]
                v = triangle[(i + 1) % 3]
                pos1 = G.nodes[u]['pos']
                pos2 = G.nodes[v]['pos']
                distance = math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
                if G.nodes[u]['type'] == 'endnode' and G.nodes[v]['type'] == 'repeater':
                    if distance <= l_er and G.degree(u) < 3:
                        G.add_edge(u, v, type='endnode', dis=distance)
                if G.nodes[u]['type'] == 'repeater' and G.nodes[v]['type'] == 'repeater':
                    if distance <= l_rr:
                        G.add_edge(u, v, type='repeater', dis=distance)
        ##-graph_plot(G)

        # Do Delaunay Triangulation for all repeater nodes:
        # Get all repeater nodes
        repeater_positions = []
        for node in G.nodes:
            if G.nodes[node]['type'] == 'repeater':
                repeater_positions.append(G.nodes[node]['pos'])
        # Do Delaunay Triangulation
        points = np.array(repeater_positions)
        tri = Delaunay(points)
        endnode_num = len([node for node in G.nodes if G.nodes[node]['type'] == 'endnode'])
        # Add edges to G
        for triangle in tri.simplices:
            for i in range(3):
                u = triangle[i] + endnode_num
                v = triangle[(i + 1) % 3] + endnode_num
                pos1 = G.nodes[u]['pos']
                pos2 = G.nodes[v]['pos']
                distance = math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
                if G.nodes[u]['type'] == 'repeater' and G.nodes[v]['type'] == 'repeater':
                    if distance <= l_rr:
                        G.add_edge(u, v, type='repeater', dis=distance)

                # plot the graph
        ##-print("After do Delaunay Triangulation for all repeater nodes:")
        ##-graph_plot(G)
        for node in G.nodes:
            if G.nodes[node]['type'] == 'endnode' and G.degree(node) == 0:
                ##-print("endnode not connected to any repeater")
                # connect this endnode to 2-nearest repeaters
                endnode_pos = G.nodes[node]['pos']
                nearset_num = 2
                candi = []
                for r in G.nodes:
                    if G.nodes[r]['type'] == 'repeater' and abs(G.nodes[r]['pos'][0] - endnode_pos[0]) <= 2 * self.step_size and abs(G.nodes[r]['pos'][1] - endnode_pos[1]) <= 2 * step_size:
                        candi.append(r)
                candi.sort(key=lambda x: math.sqrt((G.nodes[x]['pos'][0] - endnode_pos[0]) ** 2 + (G.nodes[x]['pos'][1] - endnode_pos[1]) ** 2))
                for i in range(min(nearset_num, len(candi))):
                    G.add_edge(node, candi[i], type='endnode', dis=math.sqrt((G.nodes[candi[i]]['pos'][0] - endnode_pos[0]) ** 2 + (G.nodes[candi[i]]['pos'][1] - endnode_pos[1]) ** 2))

                


        ##-graph_plot(G)
        #Clustering
        # 获取优化后的 repeater 坐标
        repeater_positions = []
        for node in G.nodes:
            if G.nodes[node]['type'] == 'repeater':
                repeater_positions.append(G.nodes[node]['pos'])

        num_clusters = len(repeater_positions) - 3#1
        min_repeaters_num = len(repeater_positions)
        while num_clusters > 0:
            if self.is_legal(self.clustering_repeater(num_clusters, G, repeater_positions)):
                if num_clusters < min_repeaters_num:
                    min_repeaters_num = num_clusters
            num_clusters -= 4#1

        P = self.construct_new_graph(G, min_repeaters_num)

        ############################
        P_temp = P
        P_best = P
        min_repeaters_num_temp = min_repeaters_num
        repeater_positions_temp = []
        while self.is_legal(P_temp):
            ##-print("it works for min_repeaters_num_temp:", min_repeaters_num_temp)
            P_best = P_temp
            min_repeaters_num_temp -= 8#1
            for node in P_temp.nodes:
                if P_temp.nodes[node]['type'] == 'repeater':
                    repeater_positions_temp.append(P_temp.nodes[node]['pos'])
            #P_temp = clustering_repeater(min_repeaters_num_temp, P_temp, repeater_positions_temp)
            P_temp = self.construct_new_graph(P_temp, min_repeaters_num_temp)
            self.optimize_repeater_pos_light(P_temp, [node for node in P_temp.nodes if P_temp.nodes[node]['type'] == 'endnode'], l_er, l_rr)
            ##-graph_statistics(P_temp)
            ##-self.plot_and_metrics(P_temp)
            ##-print("for fine granularity, min_repeaters_num_temp:", min_repeaters_num_temp)


        G = P_best
        #########################
        # G = P

        # print the number of repeaters
        ##-print("The number of repeaters in start graph:", len([node for node in G.nodes if G.nodes[node]['type'] == 'repeater']))
        ##-print("The number of endnodes in start graph:", len([node for node in G.nodes if G.nodes[node]['type'] == 'endnode']))
        # graph_plot(G)
        return G



    def is_legal(self, P):
        # 判断是否所有edge的距离都合法：不合法返回False，合法返回True
        endnodes = [node for node in P.nodes if P.nodes[node]['type'] == 'endnode']
        edges_all_legal = True
        for edge in P.edges:
            u, v = edge
            u_pos = P.nodes[u]['pos']
            v_pos = P.nodes[v]['pos']
            distance = math.sqrt(math.pow(u_pos[0] - v_pos[0], 2) + math.pow(u_pos[1] - v_pos[1], 2))
            if (u < len(endnodes)):
                if distance > l_er:
                    edges_all_legal = False
                    break
            else:
                if distance > l_rr:
                    edges_all_legal = False
                    break

        # check if all repeaters are connected using DFS
        repeaters_connected = True
        if edges_all_legal:
            repeaters = [node for node in P.nodes if P.nodes[node]['type'] == 'repeater']
            visited = [False] * len(repeaters)
            stack = []
            stack.append(repeaters[0])
            visited[0] = True
            while stack:
                s = stack.pop()
                for i in P.adj[s]:
                    if i >= len(endnodes):
                        if visited[i - len(endnodes)] == False:
                            stack.append(i)
                            visited[i - len(endnodes)] = True
            for i in range(len(visited)):
                if visited[i] == False:
                    repeaters_connected = False
                    break
        
        # demands_satisfied = True
        # if repeaters_connected and edges_all_legal:
        #     demands_satisfied = self.demands.demand_satisfying_test(P, self.demands_list)
        # demands_satisfied = self.demands.demand_satisfying_test(P, self.demands_list)

        # if not demands_satisfied:
        #     print("Demands not satisfied when repeater node number is ", P.number_of_nodes() - len(endnodes))
        return edges_all_legal and repeaters_connected # and demands_satisfied


    # 聚类，返回用最小的合法repeaters数量作为clusters数量再做一次clustering后的图
    def clustering_repeater(self, num_clusters, T, repeater_positions):
        # 执行 K-means 聚类
        kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init='auto')
        labels = kmeans.fit_predict(repeater_positions)
        # print("labels",labels)
        endnodes = [node for node in T.nodes if T.nodes[node]['type'] == 'endnode']

        # 新建一张图
        P = nx.Graph()
        # 添加endnodes
        i = 0
        for node in T.nodes:
            if T.nodes[node]['type'] == 'endnode':
                P.add_node(i, pos=T.nodes[node]['pos'], type='endnode')
                i += 1

        # 新的 repeater 节点的坐标
        new_repeater_positions = kmeans.cluster_centers_  # 获取聚类中心
        # 添加repeater
        for i in range(num_clusters):
            P.add_node(i + len(endnodes), pos=new_repeater_positions[i], type='repeater')
        # 更新edges
        for edge in T.edges:
            u, v = edge
            new_repeater_v = labels[v - len(endnodes)]
            # 更新 endnode-repeater edge
            if u < len(endnodes): # u是endnode
                P.add_edge(u, new_repeater_v + len(endnodes), type='endnode')
            # 更新 repeater-repeater edge
            else:
                new_repeater_u = labels[u - len(endnodes)]
                if (new_repeater_u != new_repeater_v):
                    P.add_edge(new_repeater_u + len(endnodes), new_repeater_v + len(endnodes), type='repeater')

        # optimize_repeater_pos_light(P, endnodes, l_er, l_rr)
        return P # is_legal(P)


    def plot_and_metrics(self, P):
        # 打印指标
        endnodes = [node for node in P.nodes if P.nodes[node]['type'] == 'endnode']
        max_distance = 0
        max_rr = 0
        max_er = 0
        best_distance = 0
        count_illegal_er = 0
        count_illegal_rr = 0
        for edge in P.edges:
            u, v = edge
            u_pos = P.nodes[u]['pos']
            v_pos = P.nodes[v]['pos']
            distance = math.sqrt(math.pow(u_pos[0] - v_pos[0], 2) + math.pow(u_pos[1] - v_pos[1], 2))
            #if u < len(endnodes):
            if P.nodes[u]['type'] == 'endnode' or P.nodes[v]['type'] == 'endnode':
                if distance > l_er:
                    count_illegal_er += 1
                if distance > max_er:
                    max_er = distance
            else:
                if distance > l_rr:
                    # num_clusters += 1
                    count_illegal_rr += 1
                    # P.remove_edge(u ,v)
                if distance > max_rr:
                    max_rr = distance
            best_distance += distance
            if (distance > max_distance):
                max_distance = distance

        print("New reapters number:", P.number_of_nodes() - len(endnodes))
        print("max edge distance:", max_distance)
        print("max endnode-repeater edge:", max_er)
        print("max repeater-repeater edge:", max_rr)
        print("Distance Sum:", best_distance)
        print("The number of illegal e-r edges:", count_illegal_er)
        print("The number of illegal r-r edges:", count_illegal_rr)

        # 可视化
        # define the color map
        color_map = {'endnode': 'blue', 'repeater': 'red'}
        # draw the graph
        pos = nx.get_node_attributes(P, 'pos')
        node_colors = [color_map[P.nodes[n]['type']] for n in P.nodes()]
        edge_colors = [color_map[P.edges[e]['type']] for e in P.edges()]
        nx.draw(P, pos, with_labels=False, node_color=node_colors, edge_color=edge_colors, width=0.5, node_size=10)
        plt.show()


    def optimize_repeater_pos(self, P, endnodes, l_er, l_rr):
        # 初始化遗传算法工具箱和问题定义
        creator.create("FitnessMin", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        def create_individual(initial_values):
            return creator.Individual(initial_values)

        def evaluate(individual, graph):
            repeaters_positions = []
            endnodes_positions = []
            for i, node in enumerate(graph.nodes()):
                if graph.nodes[node]['type'] == 'endnode':
                    endnodes_positions.append(graph.nodes[node]['pos'])
            i = 0
            while (i < len(individual) - 1):
                repeaters_positions.append((individual[i], individual[i + 1]))  # 新生成的repeater的x,y坐标
                i += 2
            total_distance = 0
            # repeater-repeater edge
            for edge in graph.edges():
                u, v = edge
                if (u >= len(endnodes)):
                    distance = math.sqrt(
                        math.pow(repeaters_positions[u - len(endnodes)][0] - repeaters_positions[v - len(endnodes)][0],
                                2) + math.pow(
                            repeaters_positions[u - len(endnodes)][1] - repeaters_positions[v - len(endnodes)][1], 2))
                    if distance > l_rr:
                        total_distance += distance * 100
                    else:
                        total_distance += distance

            # endnode-repeater edge
            for edge in graph.edges():
                u, v = edge
                if u < len(endnodes):
                    u_pos = endnodes_positions[u]
                    v_pos = repeaters_positions[v - len(endnodes)]
                    distance = math.sqrt(math.pow(u_pos[0] - v_pos[0], 2) + math.pow(u_pos[1] - v_pos[1], 2))
                    # print("e-r:",distance)
                    # 添加限制，保证endnode-repeater的距离不能超过合法值
                    if distance > l_er:
                        total_distance += distance * 100
                    else:
                        total_distance += distance

            return -total_distance

        toolbox = base.Toolbox()
        initial_values = []
        for i, node in enumerate(P.nodes()):
            if P.nodes[node]['type'] == 'repeater':
                initial_values.append(P.nodes[node]['pos'][0])
                initial_values.append(P.nodes[node]['pos'][1])
        # 初始化individual为网格上选出的repeater1二维坐标
        # toolbox.register("map", futures.map)
        # pool = multiprocessing.Pool()
        # toolbox.register("map", pool.map)
        toolbox.register("individual", create_individual, initial_values)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evaluate, graph=P)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=10, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)
        # toolbox.register("select", tools.selBest)

        population_size = 100  # 种群大小
        num_generations = 2000  # 迭代次数

        # pool = multiprocessing.Pool(processes=32)
        # toolbox.register("map", pool.map)

        population = toolbox.population(n=population_size)
        # population = [initial_values for _ in range(population_size)]

        for generation in range(num_generations):
            offspring = toolbox.select(population, k=len(population))  # 选择
            offspring = list(map(toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                # 交叉
                if random.random() < 0.8:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                # 突变
                if random.random() < 0.2:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid_individuals = [ind for ind in offspring if not ind.fitness.valid]
            fitness_values = toolbox.map(lambda ind: toolbox.evaluate(ind, graph=P), invalid_individuals)
            for ind, fit in zip(invalid_individuals, fitness_values):
                ind.fitness.values = (fit,)  # 将 fit 转换为只包含一个元素的元组

            population[:] = offspring

            best_individual = tools.selBest(population, k=1)[0]  # 根据优化目标选出来的
            best_positions = []
            index = 0
            for i, node in enumerate(P.nodes()):
                if P.nodes[node]['type'] == 'repeater':
                    best_positions.append((best_individual[index], best_individual[index + 1]))
                    P.nodes[node]['pos'] = (best_individual[index], best_individual[index + 1])
                    index += 2

        # pool.close()
        # pool.join()

    def optimize_repeater_pos_light(self, P, endnodes, l_er, l_rr):
        # 初始化遗传算法工具箱和问题定义
        creator.create("FitnessMin", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        def create_individual(initial_values):
            return creator.Individual(initial_values)

        def evaluate(individual, graph):
            repeaters_positions = []
            endnodes_positions = []
            for i, node in enumerate(graph.nodes()):
                if graph.nodes[node]['type'] == 'endnode':
                    endnodes_positions.append(graph.nodes[node]['pos'])
            i = 0
            while (i < len(individual) - 1):
                repeaters_positions.append((individual[i], individual[i + 1]))  # 新生成的repeater的x,y坐标
                i += 2
            total_distance = 0
            # repeater-repeater edge
            for edge in graph.edges():
                u, v = edge
                if (u >= len(endnodes)):
                    distance = math.sqrt(
                        math.pow(repeaters_positions[u - len(endnodes)][0] - repeaters_positions[v - len(endnodes)][0],
                                2) + math.pow(
                            repeaters_positions[u - len(endnodes)][1] - repeaters_positions[v - len(endnodes)][1], 2))
                    if distance > l_rr:
                        total_distance += distance * 100
                    else:
                        total_distance += distance

            # endnode-repeater edge
            for edge in graph.edges():
                u, v = edge
                if u < len(endnodes):
                    u_pos = endnodes_positions[u]
                    v_pos = repeaters_positions[v - len(endnodes)]
                    distance = math.sqrt(math.pow(u_pos[0] - v_pos[0], 2) + math.pow(u_pos[1] - v_pos[1], 2))
                    # print("e-r:",distance)
                    # 添加限制，保证endnode-repeater的距离不能超过合法值
                    if distance > l_er:
                        total_distance += distance * 100
                    else:
                        total_distance += distance

            return -total_distance

        toolbox = base.Toolbox()
        initial_values = []
        for i, node in enumerate(P.nodes()):
            if P.nodes[node]['type'] == 'repeater':
                initial_values.append(P.nodes[node]['pos'][0])
                initial_values.append(P.nodes[node]['pos'][1])
        # 初始化individual为网格上选出的repeater1二维坐标
        # toolbox.register("map", futures.map)
        # pool = multiprocessing.Pool()
        # toolbox.register("map", pool.map)
        toolbox.register("individual", create_individual, initial_values)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evaluate, graph=P)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=10, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)
        # toolbox.register("select", tools.selBest)

        population_size = 100  # 种群大小
        num_generations = 400  # 迭代次数

        # pool = multiprocessing.Pool(processes=32)
        # toolbox.register("map", pool.map)
        population = toolbox.population(n=population_size)
        # population = [initial_values for _ in range(population_size)]

        for generation in range(num_generations): 
            offspring = toolbox.select(population, k=len(population))  # 选择
            offspring = list(map(toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                # 交叉
                if random.random() < 0.8:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                # 突变
                if random.random() < 0.2:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid_individuals = [ind for ind in offspring if not ind.fitness.valid]
            fitness_values = toolbox.map(lambda ind: toolbox.evaluate(ind, graph=P), invalid_individuals)
            for ind, fit in zip(invalid_individuals, fitness_values):
                ind.fitness.values = (fit,)  # 将 fit 转换为只包含一个元素的元组

            population[:] = offspring

            best_individual = tools.selBest(population, k=1)[0]  # 根据优化目标选出来的
            best_positions = []
            index = 0
            for i, node in enumerate(P.nodes()):
                if P.nodes[node]['type'] == 'repeater':
                    best_positions.append((best_individual[index], best_individual[index + 1]))
                    P.nodes[node]['pos'] = (best_individual[index], best_individual[index + 1])
                    index += 2

        # pool.close()
        # pool.join()



    # construct a new graph with `num_clusters` repeaters
    def construct_new_graph(self, G, num_clusters):
        repeater_positions = []
        for node in G.nodes:
            if G.nodes[node]['type'] == 'repeater':
                repeater_positions.append(G.nodes[node]['pos'])

        # 用最小的合法repeaters数量作为clusters数量再做一次clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init='auto')
        labels = kmeans.fit_predict(repeater_positions)
        # 新的 repeater 节点的坐标
        new_repeater_positions = kmeans.cluster_centers_  # 获取聚类中心
        P = nx.Graph()
        # 添加endnodes
        i = 0  # endnodes的数量
        for node in G.nodes:
            if G.nodes[node]['type'] == 'endnode':
                P.add_node(i, pos=G.nodes[node]['pos'], num_qubits=G.nodes[node]['num_qubits'], type='endnode')
                i += 1
        # 添加repeaters
        for j in range(num_clusters):
            P.add_node(j + i, pos=new_repeater_positions[j], type='repeater')
        # 更新edges
        for edge in G.edges:
            u, v = edge
            new_repeater_v = labels[v - i]
            # Convert new_repeater_v type from numpy.int32 to int
            new_repeater_v = int(new_repeater_v)
            # 更新 endnode-repeater edge
            if (u < i):
                P.add_edge(u, new_repeater_v + i, type='endnode')
            # 更新 repeater-repeater edge
            else:
                new_repeater_u = labels[u - i]
                # Convert new_repeater_u type from numpy.int32 to int
                new_repeater_u = int(new_repeater_u)
                if (new_repeater_u != new_repeater_v):
                    P.add_edge(new_repeater_u + i, new_repeater_v + i, type='repeater')

        return P


    def endnodes_convex_hull(self, G, endnodes):
        endnodes_positions = []
        for node in endnodes:
            endnodes_positions.append(G.nodes[node]['pos'])
        endnodes_positions = np.array(endnodes_positions)
        hull = ConvexHull(endnodes_positions)
        return hull.vertices
    
    def remove_repeaters_not_included_in_convex_hull(self, G, endnodes):
        # Remove repeaters not included in the convex hull
        convex_hull = self.endnodes_convex_hull(G, endnodes)
        repeaters = [node for node in G.nodes if G.nodes[node]['type'] == 'repeater']
        for repeater in repeaters:
            pos = G.nodes[repeater]['pos']
            # if the repeater node's coordinate is not in the convex hull range, remove it
            inside = True
            for i in range(len(convex_hull)):
                p1 = G.nodes[endnodes[convex_hull[i]]]['pos']
                p2 = G.nodes[endnodes[convex_hull[(i + 1) % len(convex_hull)]]]['pos']
                if (p1[0] - pos[0]) * (p2[1] - pos[1]) - (p1[1] - pos[1]) * (p2[0] - pos[0]) < 0:
                    inside = False
                    break
            if not inside:
                G.remove_node(repeater)
        return G
    
    def endnodes_concave_hull(self, G, endnodes):
        endnodes_positions = []
        for node in endnodes:
            endnodes_positions.append(G.nodes[node]['pos'])
        endnodes_positions = np.array(endnodes_positions)
        hull = concave_hull_indexes(endnodes_positions, concavity=12)
        print("concave hull:", hull)
        return hull

    def remove_repeaters_not_included_in_concave_hull(self, G, endnodes):
        concave_hull_t = self.endnodes_concave_hull(G, endnodes)
        repeaters = [node for node in G.nodes if G.nodes[node]['type'] == 'repeater']
        for repeater in repeaters:
            pos = G.nodes[repeater]['pos']
            # if the repeater node's coordinate is not in the convex hull range, remove it
            inside = True
            for i in range(len(concave_hull_t)):
                p1 = G.nodes[endnodes[concave_hull_t[i]]]['pos']
                p2 = G.nodes[endnodes[concave_hull_t[(i + 1) % len(concave_hull_t)]]]['pos']
                if (p1[0] - pos[0]) * (p2[1] - pos[1]) - (p1[1] - pos[1]) * (p2[0] - pos[0]) < 0:
                    inside = False
                    break
            if not inside:
                G.remove_node(repeater)
        return G

    def copy_graph_with_reindex(self, G):
        P = nx.Graph()
        node_count = 0
        node_map = {}
        for node in G.nodes:
            if G.nodes[node]['type'] == 'endnode':
                P.add_node(node_count, pos=G.nodes[node]['pos'], num_qubits=G.nodes[node]['num_qubits'], type='endnode')
            else:
                P.add_node(node_count, pos=G.nodes[node]['pos'], type='repeater')
            node_map[node] = node_count
            node_count += 1
        for edge in G.edges:
            u, v = edge
            P.add_edge(node_map[u], node_map[v], type=G.edges[edge]['type'], dis=G.edges[edge]['dis'])
        return P
    
    def optimize_repeater_pos_t(self, P, endnodes, l_er, l_rr):
        # 初始化遗传算法工具箱和问题定义
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        def create_individual(initial_values):
            return creator.Individual(initial_values)
        """
        def evaluate(individual, graph):
            repeaters_positions = []
            endnodes_positions = []
            for i, node in enumerate(graph.nodes()):
                if graph.nodes[node]['type'] == 'endnode':
                    endnodes_positions.append(graph.nodes[node]['pos'])
            i = 0
            while (i < len(individual) - 1):
                repeaters_positions.append((individual[i], individual[i + 1]))  # 新生成的repeater的x,y坐标
                i += 2

            distances = []
            total_penalty = 0

            # repeater-repeater edge
            for edge in graph.edges():
                u, v = edge
                if (u >= len(endnodes)) and (v >= len(endnodes)):
                    distance = math.sqrt(
                        math.pow(repeaters_positions[u - len(endnodes)][0] - repeaters_positions[v - len(endnodes)][0], 2) + 
                        math.pow(repeaters_positions[u - len(endnodes)][1] - repeaters_positions[v - len(endnodes)][1], 2))
                    if distance > l_rr:
                        total_penalty += (distance - l_rr) ** 2  # 添加对超长边的惩罚
                    distances.append(distance)

            # endnode-repeater edge
            for edge in graph.edges():
                u, v = edge
                if u < len(endnodes):
                    u_pos = endnodes_positions[u]
                    v_pos = repeaters_positions[v - len(endnodes)]
                    distance = math.sqrt(math.pow(u_pos[0] - v_pos[0], 2) + math.pow(u_pos[1] - v_pos[1], 2))
                    if distance > l_er:
                        total_penalty += (distance - l_er) ** 2  # 添加对超长边的惩罚
                    distances.append(distance)

            mean_distance = sum(distances) / len(distances)
            variance = sum((d - mean_distance) ** 2 for d in distances) / len(distances)
            total_variance = variance + total_penalty
            return (total_variance,)
            
        """
        def evaluate(individual, graph):
            repeaters_positions = []
            endnodes_positions = []
            for i, node in enumerate(graph.nodes()):
                if graph.nodes[node]['type'] == 'endnode':
                    endnodes_positions.append(graph.nodes[node]['pos'])
            i = 0
            while i < len(individual) - 1:
                repeaters_positions.append((individual[i], individual[i + 1]))  # 新生成的repeater的x,y坐标
                i += 2

            distances = []
            total_penalty = 0

            # 计算地图中心
            all_positions = endnodes_positions# + repeaters_positions
            center_x = sum(pos[0] for pos in all_positions) / len(all_positions) #map_size / 2#
            center_y = sum(pos[1] for pos in all_positions) / len(all_positions) #map_size / 2 #

            # repeater-repeater edge
            for edge in graph.edges():
                u, v = edge
                if u >= len(endnodes) and v >= len(endnodes):
                    distance = math.sqrt(
                        math.pow(repeaters_positions[u - len(endnodes)][0] - repeaters_positions[v - len(endnodes)][0], 2) +
                        math.pow(repeaters_positions[u - len(endnodes)][1] - repeaters_positions[v - len(endnodes)][1], 2))
                    if distance > l_rr:
                        total_penalty += (distance - l_rr) ** 2  # 添加对超长边的惩罚
                    distances.append(distance)

            # endnode-repeater edge
            for edge in graph.edges():
                u, v = edge
                if u < len(endnodes):
                    u_pos = endnodes_positions[u]
                    v_pos = repeaters_positions[v - len(endnodes)]
                    distance = math.sqrt(math.pow(u_pos[0] - v_pos[0], 2) + math.pow(u_pos[1] - v_pos[1], 2))
                    if distance > l_er:
                        total_penalty += (distance - l_er) ** 2  # 添加对超长边的惩罚
                    distances.append(distance)

            mean_distance = sum(distances) / len(distances)
            variance = sum((d - mean_distance) ** 2 for d in distances) / len(distances)

            # 计算repeater nodes到地图中心的距离
            center_distance = sum(
                math.sqrt(math.pow(pos[0] - center_x, 2) + math.pow(pos[1] - center_y, 2))
                for pos in repeaters_positions
            ) / len(repeaters_positions)

            total_score = variance + total_penalty + center_distance
            return (total_score,)
       

        toolbox = base.Toolbox()
        initial_values = []
        for i, node in enumerate(P.nodes()):
            if P.nodes[node]['type'] == 'repeater':
                initial_values.append(P.nodes[node]['pos'][0])
                initial_values.append(P.nodes[node]['pos'][1])

        toolbox.register("individual", create_individual, initial_values)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evaluate, graph=P)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=10, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)

        population_size = 100  # 种群大小
        num_generations = 2000  # 迭代次数

        population = toolbox.population(n=population_size)

        for generation in range(num_generations):
            offspring = toolbox.select(population, k=len(population))  # 选择
            offspring = list(map(toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                # 交叉
                if random.random() < 0.8:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                # 突变
                if random.random() < 0.2:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid_individuals = [ind for ind in offspring if not ind.fitness.valid]
            fitness_values = toolbox.map(lambda ind: toolbox.evaluate(ind, graph=P), invalid_individuals)
            for ind, fit in zip(invalid_individuals, fitness_values):
                ind.fitness.values = fit  # 将 fit 转换为只包含一个元素的元组

            population[:] = offspring

            best_individual = tools.selBest(population, k=1)[0]  # 根据优化目标选出来的
            best_positions = []
            index = 0
            for i, node in enumerate(P.nodes()):
                if P.nodes[node]['type'] == 'repeater':
                    best_positions.append((best_individual[index], best_individual[index + 1]))
                    P.nodes[node]['pos'] = (best_individual[index], best_individual[index + 1])
                    index += 2


    def optimize_repeater_pos_light_t(self, P, endnodes, l_er, l_rr):
        # 初始化遗传算法工具箱和问题定义
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        def create_individual(initial_values):
            return creator.Individual(initial_values)
        """
        def evaluate(individual, graph):
            repeaters_positions = []
            endnodes_positions = []
            for i, node in enumerate(graph.nodes()):
                if graph.nodes[node]['type'] == 'endnode':
                    endnodes_positions.append(graph.nodes[node]['pos'])
            i = 0
            while (i < len(individual) - 1):
                repeaters_positions.append((individual[i], individual[i + 1]))  # 新生成的repeater的x,y坐标
                i += 2

            distances = []
            total_penalty = 0

            # repeater-repeater edge
            for edge in graph.edges():
                u, v = edge
                if (u >= len(endnodes)) and (v >= len(endnodes)):
                    distance = math.sqrt(
                        math.pow(repeaters_positions[u - len(endnodes)][0] - repeaters_positions[v - len(endnodes)][0], 2) + 
                        math.pow(repeaters_positions[u - len(endnodes)][1] - repeaters_positions[v - len(endnodes)][1], 2))
                    if distance > l_rr:
                        total_penalty += (distance - l_rr) ** 2  # 添加对超长边的惩罚
                    distances.append(distance)

            # endnode-repeater edge
            for edge in graph.edges():
                u, v = edge
                if u < len(endnodes):
                    u_pos = endnodes_positions[u]
                    v_pos = repeaters_positions[v - len(endnodes)]
                    distance = math.sqrt(math.pow(u_pos[0] - v_pos[0], 2) + math.pow(u_pos[1] - v_pos[1], 2))
                    if distance > l_er:
                        total_penalty += (distance - l_er) ** 2  # 添加对超长边的惩罚
                    distances.append(distance)

            mean_distance = sum(distances) / len(distances)
            variance = sum((d - mean_distance) ** 2 for d in distances) / len(distances)
            total_variance = variance + total_penalty
            return (total_variance,)
        """    
        def evaluate(individual, graph):
            repeaters_positions = []
            endnodes_positions = []
            for i, node in enumerate(graph.nodes()):
                if graph.nodes[node]['type'] == 'endnode':
                    endnodes_positions.append(graph.nodes[node]['pos'])
            i = 0
            while i < len(individual) - 1:
                repeaters_positions.append((individual[i], individual[i + 1]))  # 新生成的repeater的x,y坐标
                i += 2

            distances = []
            total_penalty = 0

            # 计算地图中心
            all_positions = endnodes_positions + repeaters_positions
            center_x = self.map_size / 2#sum(pos[0] for pos in all_positions) / len(all_positions)
            center_y = self.map_size / 2 #sum(pos[1] for pos in all_positions) / len(all_positions)

            # repeater-repeater edge
            for edge in graph.edges():
                u, v = edge
                if u >= len(endnodes) and v >= len(endnodes):
                    distance = math.sqrt(
                        math.pow(repeaters_positions[u - len(endnodes)][0] - repeaters_positions[v - len(endnodes)][0], 2) +
                        math.pow(repeaters_positions[u - len(endnodes)][1] - repeaters_positions[v - len(endnodes)][1], 2))
                    if distance > l_rr:
                        total_penalty += (distance - l_rr) ** 2  # 添加对超长边的惩罚
                    distances.append(distance)

            # endnode-repeater edge
            for edge in graph.edges():
                u, v = edge
                if u < len(endnodes):
                    u_pos = endnodes_positions[u]
                    v_pos = repeaters_positions[v - len(endnodes)]
                    distance = math.sqrt(math.pow(u_pos[0] - v_pos[0], 2) + math.pow(u_pos[1] - v_pos[1], 2))
                    if distance > l_er:
                        total_penalty += (distance - l_er) ** 2  # 添加对超长边的惩罚
                    distances.append(distance)

            mean_distance = sum(distances) / len(distances)
            variance = sum((d - mean_distance) ** 2 for d in distances) / len(distances)

            # 计算repeater nodes到地图中心的距离
            center_distance = sum(
                math.sqrt(math.pow(pos[0] - center_x, 2) + math.pow(pos[1] - center_y, 2))
                for pos in repeaters_positions
            ) / len(repeaters_positions)

            total_score = variance + total_penalty + center_distance
            return (total_score,)
       
        toolbox = base.Toolbox()
        initial_values = []
        for i, node in enumerate(P.nodes()):
            if P.nodes[node]['type'] == 'repeater':
                initial_values.append(P.nodes[node]['pos'][0])
                initial_values.append(P.nodes[node]['pos'][1])

        toolbox.register("individual", create_individual, initial_values)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evaluate, graph=P)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=10, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)

        population_size = 100  # 种群大小
        num_generations = 2000  # 迭代次数

        population = toolbox.population(n=population_size)

        for generation in range(num_generations):
            offspring = toolbox.select(population, k=len(population))  # 选择
            offspring = list(map(toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                # 交叉
                if random.random() < 0.8:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                # 突变
                if random.random() < 0.2:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid_individuals = [ind for ind in offspring if not ind.fitness.valid]
            fitness_values = toolbox.map(lambda ind: toolbox.evaluate(ind, graph=P), invalid_individuals)
            for ind, fit in zip(invalid_individuals, fitness_values):
                ind.fitness.values = fit  # 将 fit 转换为只包含一个元素的元组

            population[:] = offspring

            best_individual = tools.selBest(population, k=1)[0]  # 根据优化目标选出来的
            best_positions = []
            index = 0
            for i, node in enumerate(P.nodes()):
                if P.nodes[node]['type'] == 'repeater':
                    best_positions.append((best_individual[index], best_individual[index + 1]))
                    P.nodes[node]['pos'] = (best_individual[index], best_individual[index + 1])
                    index += 2

def read_endnodes_init_grid_graph_with_grid_edges_t(endnodes_graph_file, map_size=1000, grid_size=15):
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
    # print(type(map_size), type(grid_size))
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
    ##-graph_plot(G)




    # Connect endnodes to repeaters if the distance is less than l_er
    repeaters = [node for node in G.nodes if G.nodes[node]['type'] == 'repeater']

    # Return the graph and the endnodes
    return G, endnodes
