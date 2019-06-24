import sys
import numpy as np
from modules.language_model import LanguageModel
from modules.lego_model import LegoModel
from modules.softmax_cross_entropy import Softmax, SoftmaxCrossEntropy, NegativeLogLikelihood

lr = 0.1
hidden_size = 64
embedding_size = 32
batch_size  = 1
# sequence    = [2, 3, 4, 1, 1, 2, 3, 2, 2, 1, 4, 2]
seq_len     = 16
input_file = "data/input.txt"
data_size = 10000 # only reads first 10000 characters
max_epoch = 10
   
if __name__ == "__main__":

    data = open(input_file, 'r').read()
    chars = list(set(data))
    
    char_to_ix = { ch:i for i,ch in enumerate(chars) }
    ix_to_char = { i:ch for i,ch in enumerate(chars) }
    
    sequence    = [2, 3, 4, 1, 1, 2, 3, 2, 2, 1, 4, 2]
    
    vocab_size = len(chars)
    
    print("Vocab size: %d" % vocab_size)
    print("Learning rate: %.6f" % lr)
    m = LanguageModel(vocab_size, embedding_size, hidden_size, vocab_size)
    m.reset_hidden(1)
    # criterion = SoftmaxCrossEntropy(dim=2)
    criterion = NegativeLogLikelihood()
    
    # input sequences from the beginning of the data file
    index = 0
    n_updates = 0
    epoch = 0
    while True:
        if index + seq_len + 1 >= min(len(data), data_size): 
            index = 0
            epoch += 1
            m.reset_hidden(1)
            if epoch > max_epoch:
                break

        inputs = [[char_to_ix[ch] for ch in data[index:index+seq_len]]]
        targets = [[char_to_ix[ch] for ch in data[index+1:index+seq_len+1]]]
        index = index + seq_len
           
        x = np.asarray(inputs).T

        y = (np.arange(seq_len), np.arange(batch_size), x)

        print(x.shape)

        print(y)

        # y = np.asarray(targets).T

        batch_size = x.shape[1]
      
        out = m(x)

        # do forward bass and backward pass for backprop
        
        # loss = criterion(out, y)
        # ppl = np.exp(loss / (seq_len * batch_size))

        # dLdOut = criterion.backward(out, y)
        
        # m.zero_grad()
        # _ = m.backward(dLdOut)
        
        # # grad check the linear layer
        # layer = m.out
        
        # w = layer.parameters['W']
        # grad = np.copy(layer.parameters['W'])

        # print("CHECK GRAD")
        # num_checks, delta = 10, 1e-5
        
        # for i in range(w.size):
            
        #     w_ = w.flat[i]
        #     grad_backprop = grad.flat[i]
            
        #     w.flat[i] = w_ + delta
        #     m.reset_hidden(1)
        #     out = m(x)
        #     lstm_out_1 = np.copy(m.lstm_out)
        
        #     loss_1 = criterion(out, y)
            
        #     w.flat[i] = w_ - delta
        #     m.reset_hidden(1)
        #     out = m(x)
        #     lstm_out_2 = np.copy(m.lstm_out)
            
        #     loss_2 = criterion(out, y)
            
        #     check = np.sum(lstm_out_1 - lstm_out_2)
        #     print(check)
        #     grad_numerical = (loss_1 - loss_2) / ( 2 * delta )  
        #     rel_error = abs(grad_backprop - grad_numerical) / abs(grad_numerical + grad_backprop)
            
        #     print ('%f, %f => %e ' % (grad_numerical, grad_backprop, rel_error))

        
                
       
        break
           
            
            #print(w.shape)
        # print(weight.shape)
