import numpy as np
from util.util import sigmoid
from modules.module import Module
from numpy import tanh

class LSTMLayer(Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        std = 1.0 / np.sqrt(input_size + hidden_size)

        self.W_hh = np.random.normal(0, std, [hidden_size, 4*hidden_size]).astype(np.float32)
        self.W_ih = np.random.normal(0, std, [input_size,  4*hidden_size]).astype(np.float32)

    def forward(self, x_t, (h_in, c_in)):
        y_t = np.dot(x_t, self.W_ih) + np.dot(h_in, self.W_hh)
        i_t, f_t, g_t, o_t = np.split(y_t, 4, axis=1)

        i_t = sigmoid(i_t)
        f_t = sigmoid(f_t)
        g_t = tanh(g_t)
        o_t = sigmoid(o_t)

        c_out = f_t * c_in + i_t * g_t
        tanh_c = tanh(c_out)
        h_out = o_t * tanh_c

        activations = tanh_c, o_t, g_t, f_t, i_t, h_in, c_in, x_t
        return h_out, c_out, activations

    def backward(self, activations, dLdOut):
        '''
        Differentiate layer outputs w.r.t. weights and w.r.t. inputs.
        '''
        tanh_c_out, o_t, g_t, f_t, i_t, h_in, c_in, x_t = activations
        dLdhout, dLdcout_part = dLdOut

        dhoutdo = tanh_c_out
        # dLdcout_part = dL/dc_{t+1} * dc_{t+1}/dc_{t} = dL/dc_{t+1} * f_{t+1}
        dLdcout = dLdhout * o_t * (1 - np.square(tanh_c_out)) + dLdcout_part

        didy = i_t * (1 - i_t)
        dfdy = f_t * (1 - f_t)
        dgdy = 1 - np.square(g_t)
        dody = o_t * (1 - o_t)

        dcoutdf = c_in
        dcoutdg = i_t
        dcoutdi = g_t

        dLdy = np.concatenate([dLdcout * dcoutdi * didy,
                               dLdcout * dcoutdf * dfdy,
                               dLdcout * dcoutdg * dgdy,
                               dLdhout * dhoutdo * dody], 1)

        self.gradients["W_ih"] += np.dot(x_t.transpose(),  dLdy)
        self.gradients["W_hh"] += np.dot(h_in.transpose(), dLdy)

        dLdhin = np.dot(dLdy, self.W_hh.transpose())
        dLdx   = np.dot(dLdy, self.W_ih.transpose())

        dcoutdcin = f_t
        dLdcin = dLdcout * dcoutdcin

        return dLdx, (dLdhin, dLdcin)
