
# Generate the grid network
map_size = 5000#1000
grid_size = 17#15
step_size = map_size / grid_size
l_rr = 400#step_size * 4
l_er = 400#step_size * 3


max_single_dis = 150#step_size * 3

unit_fiber_cost = 1
unit_repeater_cost = 1000

link_failure_rate = [0.01, 0.02, 0.04, 0.08, 0.1]

c_fiber = 200000.0