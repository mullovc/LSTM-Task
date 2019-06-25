import sys
import numpy as np
from modules.language_model import LanguageModel
from modules.lego_model import LegoModel
from modules.softmax_cross_entropy import SoftmaxCrossEntropy

lr = 0.01
embedding_size = 16
hidden_size    = 16
batch_size     = 64
sequence       = [2, 3, 4, 1, 1, 2, 3, 2, 2, 1, 4, 2]
seq_len        = len(sequence)
input_size     = max(sequence) + 1
output_size    = input_size

x = np.expand_dims(np.array([0] + sequence[:-1]), 1).repeat(batch_size, 1)

# y = (np.arange(seq_len), np.arange(batch_size), sequence)
seq_idx = np.arange(seq_len).repeat(batch_size, 0)
batch_idx = np.expand_dims(np.arange(batch_size), 0).repeat(seq_len, 0).flatten()
tgt_idx = np.array(sequence).repeat(batch_size, 0)
y = (seq_idx, batch_idx, tgt_idx)


m = LanguageModel(input_size, embedding_size, hidden_size, output_size)
criterion = SoftmaxCrossEntropy(dim=2)

def print_pred(pred, tgt):
    # this cryptic code is responsible for colored in-place output of the
    # prediction
    green  = "\x1b[32m"
    red    = "\x1b[31m"
    normal = "\x1b[37m"

    buf = " ".join(["{}{: 2}".format(green if p == t else red, p) for p, t in zip(pred, tgt)])
    sys.stdout.write("\r" + buf + normal)

for i in range(1000):
    out = m(x)
    loss = criterion(out, y)

    m.zero_grad()
    dLdOut = criterion.backward()
    _ = m.backward(dLdOut)

    m.apply_gradient(lr)

    pred = np.argmax(out, 2)

    for p in pred.transpose():
        print_pred(p.flatten(), sequence)
