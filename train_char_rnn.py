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


# #x = np.eye(input_size)[[0] + sequence[:-1]].reshape([seq_len, 1, input_size])
# x = np.array([0] + sequence[:-1]).reshape([seq_len, 1])

# y = (np.arange(seq_len), np.arange(batch_size), sequence)


# m = LanguageModel(input_size, embedding_size, hidden_size, output_size)
# criterion = SoftmaxCrossEntropy(dim=2)
# softmax = Softmax(dim=2)

# def print_pred(pred, tgt):
    # # this cryptic code is responsible for colored in-place output of the
    # # prediction
    # green  = "\x1b[32m"
    # red    = "\x1b[31m"
    # normal = "\x1b[37m"

    # buf = " ".join(["{}{: 2}".format(green if p == t else red, p) for p, t in zip(pred, tgt)])
    # sys.stdout.write("\r" + buf + normal)

# for i in range(10000):
    
    # print(i)
    
    # m.reset_hidden(batch_size)
    # out = m(x)
    # loss = criterion(out, y)

    # m.zero_grad()
    # dLdOut = criterion.backward()
    # _ = m.backward(dLdOut)

    # m.apply_gradient(lr)

    # pred = np.argmax(out, 2)
    # print_pred(pred.flatten(), sequence)
   
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
        y = np.asarray(targets).T
        batch_size = x.shape[1]
      
        out = m(x)
        
        loss = criterion(out, y)
        ppl = np.exp(loss / (seq_len * batch_size))

        dLdOut = criterion.backward(out, y)
        
        
        m.zero_grad()
        _ = m.backward(dLdOut)
        
        # m.apply_gradient(lr)
        n_updates += 1
        
        # grad check
        layer = m.out
        
        w = layer.parameters['W']
        grad = np.copy(layer.parameters['W'])

        # print("CHECK GRAD")
        # num_checks, delta = 10, 1e-5
        
        # for i in range(w.size):
            
            # w_ = w.flat[i]
            # grad_backprop = grad.flat[i]
            
            # w.flat[i] = w_ + delta
            # out = m(x)
            # lstm_out_1 = np.copy(m.lstm_out)
        
            # loss_1 = criterion(out, y)
            
            # w.flat[i] = w_ - delta
            # out = m(x)
            # lstm_out_2 = np.copy(m.lstm_out)
            
            # loss_2 = criterion(out, y)
            
            # check = np.sum(lstm_out_1 - lstm_out_2)
            # grad_numerical = (loss_1 - loss_2) / ( 2 * delta )  
            # rel_error = abs(grad_backprop - grad_numerical) / abs(grad_numerical + grad_backprop)
            
            # print ('%f, %f => %e ' % (grad_numerical, grad_backprop, rel_error))

        # # for i in range(w.shape[0]):
            # # for j in range(w.shape[1]):
            
                # # w_ = w[i][j]
                # # grad_ = grad[i][j]
                
                # # old_val = w_
                
        m.apply_gradient(lr)    
        n_updates += 1
        
        if n_updates % 100 == 0:
            print("Epoch %d | Perplexity : %.4f" % (epoch, ppl))
            
           
            
            #print(w.shape)
        # print(weight.shape)
