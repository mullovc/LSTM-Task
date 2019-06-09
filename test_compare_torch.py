import torch
from torch.autograd import Variable
import numpy as np
from modules.LSTM import LSTM

EPSILON = 10e-7

input_size = 16
hidden_size = 8
seq_len = 10
batch_size = 1
layers = 1

X = np.random.randn(batch_size, seq_len, input_size).astype(np.float32)
h_0 = np.random.randn(hidden_size, batch_size).astype(np.float32)
c_0 = np.random.randn(hidden_size, batch_size).astype(np.float32)

X_torch = Variable(torch.from_numpy(X))
h_0_torch = Variable(torch.from_numpy(h_0).t().unsqueeze(0))
c_0_torch = Variable(torch.from_numpy(c_0).t().unsqueeze(0))

lstm = LSTM(input_size, hidden_size)

W_ih = torch.nn.Parameter(torch.from_numpy(lstm.W_ih.astype(np.float32)))
W_hh = torch.nn.Parameter(torch.from_numpy(lstm.W_hh.astype(np.float32)))

lstm_torch = torch.nn.LSTM(input_size, hidden_size, layers, batch_first=True, bias=False)
lstm_torch.weight_ih_l0 = W_ih
lstm_torch.weight_hh_l0 = W_hh

output, (h_n, c_n) = lstm(X, (h_0, c_0))
ref_output, (h_ref, c_ref) = lstm_torch(X_torch, (h_0_torch, c_0_torch))

ref_output_np = ref_output.data.numpy()
h_ref_np = h_ref.data.numpy()
c_ref_np = c_ref.data.numpy()

is_equal = np.abs(output - ref_output_np) < EPSILON

if is_equal.all():
    print("output is equal to reference")
