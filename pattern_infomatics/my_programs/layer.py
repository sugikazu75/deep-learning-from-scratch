import numpy as np
from function import *

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None 
        self.db = None
    def forward(self, x):
        self.x = x                           # x=input
        return np.dot(x, self.W) + self.b    # return output. dim of b will be expanded
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)          # dE/dx. for backpropagation
        self.dW = np.dot(self.x.T, dout)     # dE/dW. for renew parameter
        self.db = np.sum(dout, axis=0)       # sum of same col for batch training. for renew parameter
        return dx


class ReLU:
    def __init__(self):
        self.x = None       # input
    def forward(self, x):
        self.x = x
        out = relu(self.x)
        return out          # output ReLU(x)
    def backward(self, dout):
        # same shape
        dx = dout * relu_gradient(self.x)   # back propagation. dout * grradient(x)
        return dx


# last layer
class Softmax_cross_entripy_layer:
    def __init__(self):
        # y.shape = t.shape = (batch, output)
        self.y = None
        self.t = None
        self.loss = None
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy(self.y, self.t)
        return self.loss
    def backward(self, dout=1):
        dx = (self.y - self.t) / self.y.shape[0]
        return dx

# if __name__ == "__main__":
#     W = np.random.randn(10, 5)
#     b = np.zeros(5)
#     a = Affine(W,b)
#     print(isinstance(a, Affine))
