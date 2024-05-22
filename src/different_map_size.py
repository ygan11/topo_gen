from endnodes_gen import endnode_graph_gen

from iter_kms_ga_differ_map_size import KmsGaDms
from grid_steiner_differ_map_size import construct_steiner_tree_different_map_size
from waxman_differ_map_size import construct_waxman_different_map_size
import os
from demand_gen import Demand
from config import abs_file_path

from multiprocessing import Pool



map_size_grid_size_map = {500: 10, 1000: 15, 1500: 16, 2000: 17, 2500: 19}
# 假设这是你的处理函数
def process_file(f):
    map_size = int(f.split('-')[1])

    construct_steiner_tree_different_map_size(f, map_size=map_size, grid_size=map_size_grid_size_map[map_size])
    print ("steiner tree generated for " + f)
    print("f: ", f)
    kms = KmsGaDms()
    kms.iterate_kms_ga(f, map_size=map_size, grid_size=map_size_grid_size_map[map_size])
    print ("kms ga generated")

    construct_waxman_different_map_size(f, 3, 1000)
    print("Waxman graph generated for", f)


if __name__ == '__main__':
    topo_num = 1
    # endnode_nums = [100, 25, 50, 200, 400]
    map_sizes = [500, 1000, 1500]
    endnodes_files = []

    # # 生成文件列表
    # for i in range(topo_num):
    #     for endnode_num in endnode_nums:
    #         f = "./dist/endnodes/endnodesLocs-8-0.json" #endnode_graph_gen(endnode_num, i)
    #         endnodes_files.append(f)
    # load all files in the directory
    topo_name = ["deepPlace", "steiner", "waxman"]#, 'mca']
    for map_size in map_sizes:
      for root, dirs, files in os.walk(abs_file_path + "/dist/endnodes/map_size"):
          for file in files:
              # only format: endnodesLocs-map_size-100-y.json
              if file.endswith(".json") and file.startswith("endnodesLocs") and file.split('-')[2] == "100" and file.split('-')[1] == str(map_size):

                  for name in topo_name:
                      
                      if os.path.exists(abs_file_path + "/dist/topos/map_size/" + name + "-" + file.split('-')[1] + "-" + file.split('-')[2] + "-" + file.split('-')[3]):
                          continue
                      else:
                          print(abs_file_path + "/dist/topos/map_size/" + name + "-" + file.split('-')[1] + "-" + file.split('-')[2] + "-" + file.split('-')[3])
                          endnodes_files.append(os.path.join(root, file))
                          break
                  # if corresponding json file is found in "/dist/topos/map_size", then skip append this file

                  # endnodes_files.append(os.path.join(root, file))
    
    # endnodes_files append all files like "endnodesLocs-x-100-y.json" in the directory

        
        

    print("number of endnodes_files is ", len(endnodes_files))
    # print all files
    # for f in endnodes_files:
    #     print("f: ", f)
    #     map_size = f.split('-')[1]
    #     #convert the map_size to int
    #     map_size = int(map_size)

    #     construct_steiner_tree_different_map_size(f, map_size=map_size, grid_size=map_size_grid_size_map[map_size])
    #     print ("steiner tree generated for " + f)
        # kms = KmsGaDms()
        # kms.iterate_kms_ga(f, map_size=map_size, grid_size=map_size_grid_size_map[map_size])

    with Pool(processes=len(endnodes_files)) as pool:
        # Map process_file function to each file in the list
        pool.map(process_file, endnodes_files)

