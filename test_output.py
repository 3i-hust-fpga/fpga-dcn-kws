import torch

import numpy as np

sim_str = open('project_2/solution3/csim/build/out_sim.txt').read()
print(sim_str)
exit()
hw_str = ''

for line in open('out_hw.txt').read().strip().split('\n'):
    line = line.strip()
    parts = line.split(' ')
    if parts[0].isnumeric():
        # print(line)
        hw_str += line + '\n'
# exit()
print(hw_str)
# exit()
print(len(sim_str), len(hw_str))
line = 0
errors = []

flag = True
for i in range(len(hw_str.split('\n'))):
    # print(hw_str.split('\n')[i], sim_str.split('\n')[i])
    if hw_str.split('\n')[i] != sim_str.split('\n')[i]:
        errors.append(i)

errors = list(set(errors))
print(len(errors))

for line in errors:
    tmp_gold = float(hw_str.split('\n')[line].split(' ')[-1])

    tmp_sim = float(sim_str.split('\n')[line].split(' ')[-1])
    if abs(tmp_gold - tmp_sim) > 1e-5:
        print(line, ' ----------> hw:', tmp_gold, ' ----------> sim:', tmp_sim)
        flag = False
print('output:', flag)
exit()