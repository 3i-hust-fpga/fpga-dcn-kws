import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tvdcn import DeformConv2d 
from torch.optim import Adam

def recur(x, indent=''):
    rep = '{'
    indent += 4 * ' '

    if x.ndim == 1:
        for i in range(x.size(0)):
            rep += str(x[i].item())
            if i < x.size(0) - 1:
                rep += ', '
    else:
        rep += '\n' + indent
        for i in range(x.size(0)):
            rep += recur(x[i], indent)
            if i < x.size(0) - 1:
                rep += ',\n' + indent
        rep += '\n' + indent[:-4]
    rep += '}'
    return rep


def dump(name, dtype, weight, const=False):
    if const:
        rep = 'const ' + dtype + ' ' + name + ' = '
    else:
        rep = dtype + ' ' + name + ' = '

    rep += recur(weight)
    rep += ';\n\n'
    return rep

def softmax_2(x):
    out = torch.zeros(x.shape)
    idx = torch.zeros(x.shape[0])

    cache = x

    for i in range(cache.shape[1]):
        denom = 0
        for j in range(cache.shape[0]):
            idx[j] = cache[j][i].exp()
            denom += idx[j]
        for j in range(cache.shape[0]):
            out[j][i] = idx[j] / float(denom)

    return out

torch.random.manual_seed(1)

CONV_IN_SIZE_0 = 100
CONV_IN_SIZE_1 = 40
CONV_KERNEL_SIZE_0 = 3
CONV_KERNEL_SIZE_1 = 3
CONV_STRIDE = 1
CONV_FILTERS = 8
CONV_OFFSET_FILTERS = (2 * CONV_KERNEL_SIZE_0 * CONV_KERNEL_SIZE_1) 
CONV_OUT_SIZE_0 = (CONV_IN_SIZE_0 - CONV_KERNEL_SIZE_0 + 1)
CONV_OUT_SIZE_1 = (CONV_IN_SIZE_1 - CONV_KERNEL_SIZE_1 + 1)

GRU_HIDDEN_SIZE = 32
GRU_IN_SIZE = (CONV_OUT_SIZE_1 * CONV_FILTERS)
GRU_G_SIZE = (3 * GRU_HIDDEN_SIZE)
GRU_W_IH_SIZE = (3 * GRU_HIDDEN_SIZE * GRU_IN_SIZE)
GRU_B_IH_SIZE = (3 * GRU_HIDDEN_SIZE)
GRU_W_HH_SIZE = (3 * GRU_HIDDEN_SIZE * GRU_HIDDEN_SIZE)
GRU_B_HH_SIZE = (3 * GRU_HIDDEN_SIZE)

ATTN_SIZE = 32

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.dcn = DeformConv2d(1, CONV_FILTERS, kernel_size=(CONV_KERNEL_SIZE_0, CONV_KERNEL_SIZE_1))
        self.gru = nn.GRU(GRU_IN_SIZE, GRU_HIDDEN_SIZE)
        self.linear_1 = nn.Linear(GRU_HIDDEN_SIZE, ATTN_SIZE)
        self.linear_2 = nn.Linear(ATTN_SIZE, GRU_HIDDEN_SIZE)
        self.linear_3 = nn.Linear(GRU_HIDDEN_SIZE, 2)

    def forward(self, x):
        x, offset, w_out, b_out = self.dcn(x)
        out_dcn = x
        x = x.permute(2, 0, 3, 1)
        x = x.reshape(x.size(0), x.size(1), -1)
        x = self.gru(x)[0]
        out_gru = x
        enc = x.transpose(0, 1)
        x = self.linear_1(enc)
        x = x.tanh()
        x = self.linear_2(x)
        out_l2 = x
        x = x.softmax(1)
        out_sm = x
        x = torch.mul(x, enc)
        out_mul = x
        x = x.sum(dim=1)
        out_sum = x
        x = self.linear_3(x)

        return x, out_sum, out_mul, out_sm, out_l2, out_gru, out_dcn, w_out, b_out
                        
if __name__ == '__main__':

    x = torch.randn(1, CONV_IN_SIZE_0, CONV_IN_SIZE_1).unsqueeze(1)

    model = Model()

    # cnn = DeformConv2d(1, CONV_FILTERS, kernel_size=(CONV_KERNEL_SIZE_0, CONV_KERNEL_SIZE_1))
    optimizer = Adam(model.parameters(), lr=1e-4, betas=(.9, .999), eps=1e-8, weight_decay=1e-5, amsgrad=False)

    optimizer.zero_grad()
    model(x)[0].sum().backward()
    optimizer.step()
    model(x)[0].sum().backward()
    optimizer.step()

    out, out_sum, out_mul, out_sm, out_l2, out_gru, out_dcn, w_out, b_out = model(x)

    print('%.06f %.06f' % (out[0][0], out[0][1]))
    # for i in range(CONV_IN_SIZE_0):
    #     for j in range(CONV_IN_SIZE_1):
    #         print(x[0,0,i,j])
    # exit()

    if 0:
        print('--------- out sum ---------')
        print(out_sum.shape)
        # print(torch.allclose(out_sum.squeeze(0), sum_hw(out_mul.squeeze(0))))
        gold_str = ''
        row, col = out_sum.shape
        for i in range(row):
            for j in range(col):
                gold_str += ('%d %.06f\n' % (i * col + j, out_sum[i][j]))
        sim_str = open('project_1/solution8/csim/build/out_sum.txt').read()
        line = 0
        errors = []

        flag = True
        for i in range(len(gold_str)):
            if gold_str[i] == '\n':
                line += 1
            if gold_str[i] != sim_str[i]:
                errors.append(int(line))

        errors = list(set(errors))

        for line in errors:
            tmp_gold = float(gold_str.split('\n')[line].split(' ')[-1])
            tmp_sim = float(sim_str.split('\n')[line].split(' ')[-1])
            if abs(tmp_gold - tmp_sim) > 1e-5:
                print(line, ' ----------> gold:', tmp_gold, ' ----------> sim:', tmp_sim)
                flag = False
        print('sum:', flag)
        exit()

    if 0:
        print('--------- out sm + mul ---------')
        gold_str = ''
        row, col = out_mul.squeeze(0).shape
        for i in range(row):
            for j in range(col):
                # gold_str += ('%d %.06f %.06f %.06f\n' % (i * col + j, out_gru[i][0][j], out_sm[0][i][j], out_mul[0][i][j]))
                gold_str += ('%d %.06f\n' % (i * col + j, out_mul[0][i][j]))
        # print(gold_str)
        sim_str = open('project_1/solution8/csim/build/out_sm.txt').read()
        line = 0
        errors = []

        flag = True
        for i in range(len(gold_str)):
            if gold_str[i] == '\n':
                line += 1
            if gold_str[i] != sim_str[i]:
                errors.append(int(line))

        errors = list(set(errors))

        for line in errors:
            tmp_gold = float(gold_str.split('\n')[line].split(' ')[-1])
            tmp_sim = float(sim_str.split('\n')[line].split(' ')[-1])
            if abs(tmp_gold - tmp_sim) > 1e-6:
                print(line, ' ----------> gold:', tmp_gold, ' ----------> sim:', tmp_sim)
                flag = False
        print('l2:', flag)
        exit()

    if 0:
        print('--------- out dcn ---------')
        out_dcn = out_dcn.permute(2, 0, 3, 1).squeeze(1)
        out_dcn = out_dcn.reshape(out_dcn.shape[0], -1)
        gold_str = ''
        for i in range(out_dcn.shape[0]):
            for j in range(out_dcn.shape[1]):
                gold_str += ('%d %.06f\n' % (i * out_dcn.shape[1] + j, out_dcn[i][j]))
        sim_str = open('project_2/solution1/csim/build/out_sim.txt').read()
        line = 0
        errors = []

        flag = True
        for i in range(len(gold_str)):
            if gold_str[i] == '\n':
                line += 1
            if gold_str[i] != sim_str[i]:
                errors.append(int(line))

        for line in errors:
            tmp_gold = float(gold_str.split('\n')[line].split(' ')[-1])
            tmp_sim = float(sim_str.split('\n')[line].split(' ')[-1])
            if abs(tmp_gold - tmp_sim) > 1e-5:
                print(line, ' ----------> gold:', tmp_gold, ' ----------> sim:', tmp_sim)
                flag = False
        print('dcn:', flag)
        exit()

    if 0:
        print('--------- out gru ---------')
        W_IH = model.gru.weight_ih_l0.T.reshape(-1)
        W_HH = model.gru.weight_hh_l0.T.reshape(-1)
        B_IH = model.gru.bias_ih_l0
        B_HH = model.gru.bias_hh_l0

        out_dcn = out_dcn.permute(2, 0, 3, 1).squeeze(1)
        out_dcn = out_dcn.reshape(out_dcn.shape[0], -1)
        
        # out_gru = gru_hw(out_dcn, W_IH, B_IH, W_HH, B_HH, CONV_OUT_SIZE_0, GRU_IN_SIZE, GRU_HIDDEN_SIZE)
        out_gru = out_gru.squeeze(1)

        gold_str = ''

        for i in range(out_gru.shape[0]):
            for j in range(out_gru.shape[1]):
                gold_str += ('%d %.06f\n' % (i * out_gru.shape[1] + j, out_gru[i][j]))

        sim_str = open('project_1/demo/csim/build/out_sim.txt').read()
        flag = True
        line = 0
        errors = []
        for i in range(len(gold_str)):
            if gold_str[i] == '\n':
                line += 1
            if gold_str[i] != sim_str[i]:
                errors.append(int(line))

        for line in errors:
            tmp_gold = float(gold_str.split('\n')[line].split(' ')[-1])
            tmp_sim = float(sim_str.split('\n')[line].split(' ')[-1])
            if abs(tmp_gold - tmp_sim) > 1e-5:
                print(line, ' ----------> gold:', tmp_gold, ' ----------> sim:', tmp_sim)
                flag = False
        print('gru:', flag)
        exit()


    if 1:
        print('--------- gen weight ---------')

        w_offset = model.dcn.conv_offset.weight
        b_offset = model.dcn.conv_offset.bias

        W_IH = model.gru.weight_ih_l0.T.reshape(-1)
        W_HH = model.gru.weight_hh_l0.T.reshape(-1)
        B_IH = model.gru.bias_ih_l0
        B_HH = model.gru.bias_hh_l0

        L1_W = model.linear_1.weight
        L1_B = model.linear_1.bias

        L2_W = model.linear_2.weight
        L2_B = model.linear_2.bias

        L3_W = model.linear_3.weight
        L3_B = model.linear_3.bias

        print('x        ', x.shape)
        print('w_offset ', w_offset.squeeze(1).shape)
        print('b_offset ', b_offset.shape)
        print('w_out    ', w_out.squeeze(1).shape)
        print('b_out    ', b_out.shape)

        rep = ''
        rep += '#ifndef _WEIGHT_H_\n'
        rep += '#define _WEIGHT_H_\n\n'
        rep += '#include "parameters.h"\n\n'

        rep += dump('CONV_OFFSET_W[CONV_OFFSET_FILTERS][CONV_KERNEL_SIZE_0][CONV_KERNEL_SIZE_1]', 'DTYPE', w_offset.squeeze(1))
        rep += dump('CONV_OFFSET_B[CONV_OFFSET_FILTERS]', 'DTYPE', b_offset)
        rep += dump('CONV_OUT_W[CONV_FILTERS][CONV_KERNEL_SIZE_0][CONV_KERNEL_SIZE_1]', 'DTYPE', w_out.squeeze(1))
        rep += dump('CONV_OUT_B[CONV_FILTERS]', 'DTYPE', b_out)

        rep += dump('GRU_W_IH[GRU_W_IH_SIZE]', 'DTYPE', W_IH)
        rep += dump('GRU_B_IH[GRU_B_IH_SIZE]', 'DTYPE', B_IH)
        rep += dump('GRU_W_HH[GRU_W_HH_SIZE]', 'DTYPE', W_HH)
        rep += dump('GRU_B_HH[GRU_B_HH_SIZE]', 'DTYPE', B_HH)

        rep += dump('L1_W[L1_ROWS][L1_COLS]', 'DTYPE', L1_W)
        rep += dump('L1_B[L1_ROWS]', 'DTYPE', L1_B)
        rep += dump('L2_W[L2_ROWS][L2_COLS]', 'DTYPE', L2_W)
        rep += dump('L2_B[L2_ROWS]', 'DTYPE', L2_B)
        rep += dump('L3_W[L3_ROWS][L3_COLS]', 'DTYPE', L3_W)
        rep += dump('L3_B[L3_ROWS]', 'DTYPE', L3_B)


        rep += '\n#endif\n'

        with open('weight.h', 'w') as f:
            f.write(rep)


        rep = ''
        rep += '#ifndef _INPUT_H_\n'
        rep += '#define _INPUT_H_\n\n'
        rep += '#include "parameters.h"\n\n'
        rep += dump('CONV_INPUT[CONV_IN_SIZE_0 * CONV_IN_SIZE_1]', 'DTYPE', x.reshape(-1))
        rep += '\n#endif\n'

        with open('input.h', 'w') as f:
            f.write(rep)
