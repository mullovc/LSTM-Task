import numpy as np
from modules.module import Module
from modules.LSTM import LSTM

class Softmax(Module):
    def __init__(self, dim=1):
        super(Softmax, self).__init__()
        self.dim = dim

    def forward(self, x):
        exp_x = np.exp(x)
        
        output = np.divide(exp_x, np.sum(exp_x, self.dim, keepdims=True))
        return self.output

    def backward(self, dLdOut):
        
        return self.output * ( 1 - self.output)
        # raise NotImplementedError
        
class NegativeLogLikelihood(Module):

    def forward(self, x, y):    
        
        exp_x = np.exp(x)
        prob_x = np.divide(exp_x, np.sum(exp_x, 2, keepdims=True))
        
        log_prob =  np.log(prob_x)
        
        total_loss = 0
        
        for t in range(x.shape[0]):
        
            for b in range(x.shape[1]):
                
                total_loss -= log_prob[t][b][y[t][b]]
        
        return total_loss
        
    def backward(self, x, y):
    
        exp_x = np.exp(x)
        prob_x = np.divide(exp_x, np.sum(exp_x, 2, keepdims=True))
        dLdx = prob_x
        for t in range(x.shape[0]):
            for b in range(x.shape[1]):
                dLdx[t][b][y[t][b]] -= 1
        
        return dLdx 
        #return x - 1

class SoftmaxCrossEntropy(Module):
    def __init__(self, dim=1):
        super(SoftmaxCrossEntropy, self).__init__()
        self.dim = dim

    def forward(self, x, t):
        self.t = t
        
        self.exp_x = np.exp(x)  
        self.sum_exp_x = np.sum(self.exp_x, self.dim, keepdims=True)
        
        ce = -x[t] + np.log(self.sum_exp_x)
        
        return np.sum(ce)

    def backward(self):
        '''
        $ dL(softmax(x))/dx = softmax(x) - t $

        where $x$ is the input, $t$ is the target and $L$ is the cross entropy
        loss function. $t$ is a one-hot vector given as index of the one-hot
        dimension.
        '''
        dLdx = np.divide(self.exp_x, self.sum_exp_x)
        dLdx[self.t] -= 1
        return dLdx
