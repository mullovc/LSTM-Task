import numpy as np
from util.util import sigmoid
from modules.module import Module
from numpy import tanh

class LSTMLayer(Module):
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_hh = np.random.randn(4*hidden_size, hidden_size).astype(np.float32)
        self.W_ih = np.random.randn(4*hidden_size, input_size).astype(np.float32)

    def forward(self, x_t, (h_in, c_in)):
        y_t = np.dot(self.W_ih, x_t) + np.dot(self.W_hh, h_in)
        i_t, f_t, g_t, o_t = np.split(y_t, 4)

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
        Differentiate layer output w.r.t. weights and w.r.t. inputs.
        '''
        tanh_c_out, o_t, g_t, f_t, i_t, h_in, c_in, x_t = activations
        dLdhout, dLdcout = dLdOut

        W_ii, W_if, W_ig, W_io = np.split(self.W_ih, 4)
        W_hi, W_hf, W_hg, W_ho = np.split(self.W_hh, 4)


        dhoutdo = tanh_c_out
        #dhoutdcout = o_t * (1 - np.square(tanh_c_out)) # + dL/dc_{t+1} * f_{t+1} (= dc_{t+1}/dc_{t})
        dhoutdcout = o_t * (1 - np.square(tanh_c_out)) + dLdcout

        dody = o_t * (1 - o_t)
        didy = i_t * (1 - i_t)
        dgdy = 1 - np.square(g_t)
        dfdy = f_t * (1 - f_t)

        dcoutdcin = f_t
        dcoutdf   = c_in
        dcoutdg   = i_t
        dcoutdi   = g_t

        dLdW_io = np.dot(dLdhout * dhoutdo * dody, x_t.transpose())
        dLdW_ho = np.dot(dLdhout * dhoutdo * dody, h_in.transpose())
        dLdW_ii = np.dot(dLdhout * dhoutdcout * dcoutdi * didy, x_t.transpose())
        dLdW_hi = np.dot(dLdhout * dhoutdcout * dcoutdi * didy, h_in.transpose())
        dLdW_if = np.dot(dLdhout * dhoutdcout * dcoutdf * dfdy, x_t.transpose())
        dLdW_hf = np.dot(dLdhout * dhoutdcout * dcoutdf * dfdy, h_in.transpose())
        dLdW_ig = np.dot(dLdhout * dhoutdcout * dcoutdg * dgdy, x_t.transpose())
        dLdW_hg = np.dot(dLdhout * dhoutdcout * dcoutdg * dgdy, h_in.transpose())

        dLdhin  = np.dot(W_ho.transpose(), dLdhout * dhoutdo * dody)
        dLdx    = np.dot(W_io.transpose(), dLdhout * dhoutdo * dody)
        dLdhin += np.dot(W_hi.transpose(), dLdhout * dhoutdcout * dcoutdi * didy)
        dLdx   += np.dot(W_ii.transpose(), dLdhout * dhoutdcout * dcoutdi * didy)
        dLdhin += np.dot(W_hf.transpose(), dLdhout * dhoutdcout * dcoutdf * dfdy)
        dLdx   += np.dot(W_if.transpose(), dLdhout * dhoutdcout * dcoutdf * dfdy)
        dLdhin += np.dot(W_hg.transpose(), dLdhout * dhoutdcout * dcoutdg * dgdy)
        dLdx   += np.dot(W_ig.transpose(), dLdhout * dhoutdcout * dcoutdg * dgdy)

        dLdcin = dLdcout * dcoutdcin

        return (dLdhin, dLdcin), \
               (dLdW_ii, dLdW_if, dLdW_ig, dLdW_io, \
                dLdW_hi, dLdW_hf, dLdW_hg, dLdW_ho)
