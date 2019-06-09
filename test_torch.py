import torch
from torch.autograd import Variable
import numpy as np

input_size = 16
hidden_size = 8
seq_len = 10
batch_size = 1
layers = 1

X = Variable(torch.randn([batch_size, seq_len, input_size]))
h_0 = Variable(torch.randn([layers, batch_size, hidden_size]))
c_0 = Variable(torch.randn([layers, batch_size, hidden_size]))
lstm = torch.nn.LSTM(input_size, hidden_size, layers, batch_first=True, bias=False)

W_hh = lstm.weight_hh_l0
W_ih = lstm.weight_ih_l0

ref_output, (h_ref, c_ref) = lstm(X, (h_0, c_0))

output = []
gates  = []
h_n, c_n = h_0[0].t(), c_0[0].t()
for i in range(seq_len):
    x_t = X[:, i, :].t()
    y_t = torch.matmul(W_ih, x_t) + torch.matmul(W_hh, h_n)
    i_t, f_t, g_t, o_t = y_t.chunk(4)

    i_t = i_t.sigmoid()
    f_t = f_t.sigmoid()
    g_t = g_t.tanh()
    o_t = o_t.sigmoid()

    c_n = f_t * c_n + i_t * g_t
    h_n = o_t * c_n.tanh()

    output.append(h_n)
    gates.append(c_n)

output = torch.stack(output).squeeze().unsqueeze(0)
gates = torch.stack(gates).squeeze().unsqueeze(0)
