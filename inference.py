import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import time

from tvdcn import DeformConv2d 

torch.random.manual_seed(1)
torch.set_num_threads(16)
exit()

CONV_IN_SIZE_0 = 200
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
        x = x.permute(2, 0, 3, 1)
        x = x.reshape(x.size(0), x.size(1), -1)
        x = self.gru(x)[0]
        enc = x.transpose(0, 1)
        x = self.linear_1(enc)
        x = x.tanh()
        x = self.linear_2(x)
        x = x.softmax(1)
        x = torch.mul(x, enc)
        x = x.sum(dim=1)
        x = self.linear_3(x)

        return x
                        
if __name__ == '__main__':

    N_TIMES = 1000
    x = torch.randn(1, CONV_IN_SIZE_0, CONV_IN_SIZE_1).unsqueeze(1)

    model = Model()
    start = time.time()
    for i in range(N_TIMES):
        model(x)

    duration = (time.time() - start) * 1e3 / N_TIMES
    print(f'Done in: {duration:.3f} ms')