import numpy as np

def relu(x):
    ans = np.maximum(0, x)
    return ans

def relu_gradient(x):
    ans = x.copy()
    ans[ans>=0] = 1
    ans[ans<0] = 0
    return ans

# apply for each row
def softmax(x):
    row_max = np.max(x, axis=1, keepdims=True)
    e_x = np.exp(x - row_max)   # overflow  
    ans = e_x / np.sum(e_x, axis=1, keepdims=True)
    return ans

# y.shape = t.shape = (batch, output)
def cross_entropy(y, t):
    batch = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch


# if __name__ == "__main__":
#     x = np.array([[1,5,3,2], [9,4,2,6]])
#     print(softmax(x))
