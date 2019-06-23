import sys
import numpy as np
from modules.language_model import LanguageModel
from modules.lego_model import LegoModel
from modules.softmax_cross_entropy import Softmax, SoftmaxCrossEntropy

lr = 0.01
hidden_size = 32
embedding_size = 32
batch_size  = 1
sequence    = [2, 3, 4, 1, 1, 2, 3, 2, 2, 1, 4, 2]
seq_len     = len(sequence)
input_size  = max(sequence) + 1
output_size = input_size

#x = np.eye(input_size)[[0] + sequence[:-1]].reshape([seq_len, 1, input_size])
x = np.array([0] + sequence[:-1]).reshape([seq_len, 1])

y = (np.arange(seq_len), np.arange(batch_size), sequence)


m = LanguageModel(input_size, embedding_size, hidden_size, output_size)
criterion = SoftmaxCrossEntropy(dim=2)
softmax = Softmax(dim=2)

def print_pred(pred, tgt):
    # this cryptic code is responsible for colored in-place output of the
    # prediction
    green  = "\x1b[32m"
    red    = "\x1b[31m"
    normal = "\x1b[37m"

    buf = " ".join(["{}{: 2}".format(green if p == t else red, p) for p, t in zip(pred, tgt)])
    sys.stdout.write("\r" + buf + normal)

for i in range(10000):
    
    print(i)
    
    m.reset_hidden(batch_size)
    out = m(x)
    
    loss = criterion(out, y)

    m.zero_grad()
    dLdOut = criterion.backward()
    print(dLdOut.shape)
    _ = m.backward(dLdOut)

    m.apply_gradient(lr)

    pred = np.argmax(out, 2)
    print_pred(pred.flatten(), sequence)
