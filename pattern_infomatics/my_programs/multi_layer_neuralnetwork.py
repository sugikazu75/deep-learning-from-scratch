from re import X
import numpy as np
from layer import *

class Multi_Layer_NeuralNetwork:
    def __init__(self, input_size, hidden_size1, output_size):
        # init layers
        # Affine, ReLU , Affine
        affine_init_param = [[np.random.randn(input_size, hidden_size1), np.zeros(hidden_size1)],
                             [np.random.randn(hidden_size1, output_size), np.zeros(output_size)]
                             ]
        self.layers = [Affine(affine_init_param[0][0], affine_init_param[0][1]), 
                        ReLU(),
                        Affine(affine_init_param[1][0], affine_init_param[1][1]),
                        ]
        # lastlayer
        self.lastlayer = Softmax_cross_entripy_layer()

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x                                # not applied softmax

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return softmax(x)

    # t.shape = (batch, output)
    def loss(self, x, t):
        x = self.forward(x)                     # not applied softmax
        return self.lastlayer.forward(x, t)     # Softmax_cross_entropy_layer.forward(x, t) return  cross_entropy(softmax(x), t)
    
    # t.shape = (batch, output)
    def acc(self, x, t):
        y = self.forward(x)                     # not applied softmax. (not need to apply softmax)
        predict_ans = np.argmax(y, axis=1)          # predicted ans index
        ans = np.argmax(t, axis=1)              # ans index
        acc = np.sum(predict_ans == ans) / float(x.shape[0])
        return acc

    def grad_affine_params(self, x, t):
        self.loss(x, t)         # hold y, t in lastlayer to calc lastlayer.backward(dout)
        dout = self.lastlayer.backward()        # dout = 1 (default)
        for layer in reversed(self.layers): 
            dout = layer.backward(dout)
        grad = [[self.layers[0].dW, self.layers[0].db], [self.layers[2].dW, self.layers[2].db]]
        return grad