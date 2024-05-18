# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from endnodes_gen import endnode_graph_gen
# from iter_kms_ga import iterate_kms_ga
from iter_kms_ga import KmsGa
from grid_steiner import construct_steiner_tree
from waxman_topo_gen import construct_waxman
import os
from demand_gen import Demand
from config import abs_file_path


# Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     topo_num = 10
#     endnode_nums = [100, 25, 50, 200, 400]
#     endnodes_files = []
#
#     for i in range(topo_num):
#         for endnode_num in endnode_nums:
#             f = endnode_graph_gen(endnode_num, i)
#             endnodes_files.append(f)
#
#
#     for f in endnodes_files:
#         construct_steiner_tree(f)
#         print ("steiner tree generated for ", f)
#         iterate_kms_ga(f)
#         print ("kms ga generated for ", f)

        # construct_waxman(f, 5, 1000)
        # print ("waxman graph generated")


from multiprocessing import Pool


# 假设这是你的处理函数
def process_file(f):
    construct_steiner_tree(f)
    print("Steiner tree generated for", f)
    # kms = KmsGa()
    # kms.iterate_kms_ga(f)
    # print("KMS GA generated for", f)
    # construct_waxman(f, 3, 1000)
    # print("Waxman graph generated for", f)


if __name__ == '__main__':
    topo_num = 1
    endnode_nums = [100, 25, 50, 200, 400]
    endnodes_files = []

    # # 生成文件列表
    # for i in range(topo_num):
    #     for endnode_num in endnode_nums:
    #         f = "./dist/endnodes/endnodesLocs-8-0.json" #endnode_graph_gen(endnode_num, i)
    #         endnodes_files.append(f)
    # load all files in the directory
    for root, dirs, files in os.walk(abs_file_path + "/dist/endnodes"):
        for file in files:
            if file.endswith(".json"):
                endnodes_files.append(os.path.join(root, file))

    print("number of endnodes_files is ", len(endnodes_files))
    # print all files


    with Pool(processes=50) as pool:
        # Map process_file function to each file in the list
        pool.map(process_file, endnodes_files)


