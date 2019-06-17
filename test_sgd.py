import numpy as np
from modules.lego_model import LegoModel
from modules.sse import SSE
from modules.softmax_cross_entropy import Softmax, SoftmaxCrossEntropy

EPS=10e-5
lr = 0.02
input_size = 8
hidden_size = 6
output_size = 4
batch_size = 1
seq_len = 5

# determines whether to do weight updates for just the final time step, or each
# time step output
FINAL_ONLY = True

x = np.random.randn(seq_len, 1, input_size).astype(np.float32)

# Define target vectors. If doing weight updates for last time step only,
# target is a 1-D vector. If doing weight updates for each times step output,
# target is a sequence.
if FINAL_ONLY:
    # target for sigmoid output
    #y = np.array([0, 1, 0, 0], dtype=np.float32)#.reshape([output_size, 1])
    # target index for sofmax output
    y = [np.arange(batch_size), 1]
    dim = 1
else:
    # if target is a sequence, manually define a target sequence with `seq_len` 5
    y = [np.arange(seq_len), np.arange(batch_size), [0, 1, 2, 2, 3]]
    dim = 2


m = LegoModel(input_size, hidden_size, output_size, not FINAL_ONLY)
criterion = SoftmaxCrossEntropy(dim)
softmax = Softmax(dim)

for i in range(10000):
    out = m(x)
    loss = criterion(out, y)

    m.zero_grad()
    dLdOut = criterion.backward()
    _ = m.backward(dLdOut)

    m.apply_gradient(lr)

    print(softmax(out).round(1))
