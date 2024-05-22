
# Generate the grid network
map_size = 1000#1000
grid_size = 15#15
step_size = map_size / grid_size
l_rr = 200#step_size * 4
l_er = 200#step_size * 3


max_single_dis = l_er#step_size * 3

unit_fiber_cost = 1
unit_repeater_cost = 1000

link_failure_rate = [0.01, 0.02, 0.04, 0.08, 0.1]

c_fiber = 200000.0

abs_file_path = "/home/ygan11/quantum_topo_design_2024/topo_gen"

total_resource_number = 500

avg_qubit_per_repeater = 13