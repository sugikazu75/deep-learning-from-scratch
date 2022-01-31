from traceback import print_tb
import numpy as np
from two_layer_net import *
from functions import *

data_path = "heart_disease_health_indicators_BRFSS2015.csv"

all_data = np.loadtxt(data_path, dtype=float, delimiter=',')

max_data = all_data.max(axis=0)
my_data = np.array([0, 0, 0, 0, 20.7, 1, 0, 0, 1, 0, 1, 0, 0, 0, 3, 5, 0, 0, 0, 1, 6, 4])
my_data_std = np.array([my_data / max_data])
my_test = my_data_std[:, np.arange(21)+1]


def data_std(x, axis=None):
    max = x.max(axis=axis, keepdims=True)
    result = x/max
    return result

#std
all_data = data_std(all_data, 0)


# index of use data
x_index = [1, 2, 4, 5, 7, 8, 9, 10, 14, 19]
x_index = np.arange(21)+1

# divide by label
data_label0 = all_data[all_data[:, 0] == 0.0]
data_label1 = all_data[all_data[:, 0] == 1.0]

# index of use data number
label0_index = np.random.choice(data_label0.shape[0], 22000, replace=False)
label1_index = np.random.choice(data_label1.shape[0], 22000, replace=False)

# choice train and test data in the same number
train_label0 = data_label0[label0_index[np.arange(20000)], :]
test_label0 = data_label0[label0_index[np.arange(2000)+20000], :]
train_label1 = data_label1[label1_index[np.arange(20000)], :]
test_label1 = data_label1[label1_index[np.arange(2000)+20000], :]

# unify two label data
train_data = np.concatenate([train_label0, train_label1])
test_data = np.concatenate([test_label0, test_label1])

# shuffle
train_data = train_data[np.random.choice(train_data.shape[0], train_data.shape[0], replace=False), :]
test_data = test_data[np.random.choice(test_data.shape[0], test_data.shape[0], replace=False), :]

# x = input data, t = label 
train_x = train_data[:, x_index]
train_t = train_data[:, 0].astype(np.int)
test_x = test_data[:, x_index]
test_t = test_data[:, 0].astype(np.int)

def make_data():
    return (train_x, train_t), (test_x, test_t)


if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)
    print(data_label0.shape)
    print(data_label1.shape)
    print(train_data.shape)
    print(test_data.shape)
    print(train_x.shape)
    print(train_t.shape)
    print(test_x.shape)
    print(max_data)
    print(my_data_std)
    print(my_data_std.shape)
    print(my_test.shape)
    print(test_t.shape)