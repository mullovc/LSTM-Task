#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from modules.lstm_layer import LSTMLayer
from util.util import cosine

input_size = 8
hidden_size = 6
output_size = 4
batch_size = 1
seq_len = 5

x   = np.random.randn(batch_size,  input_size).astype(np.float32)
h_0 = np.random.randn(batch_size, hidden_size).astype(np.float32)
c_0 = np.random.randn(batch_size, hidden_size).astype(np.float32)

lstm_layer = LSTMLayer(input_size, hidden_size)

gradients_approx = lstm_layer.parameter_gradcheck(lambda (out, state, act): out.sum(),
                                                  x, (h_0, c_0))

out, state, activations = lstm_layer(x, (h_0, c_0))
lstm_layer.zero_grad()
lstm_layer.backward(activations, (1, 0))

gradients = lstm_layer.gradients

for p in gradients:
    print(np.mean(gradients[p] - gradients_approx[p]))
    print(cosine(gradients[p], gradients_approx[p]))
