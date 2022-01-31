import numpy as np
from function import *
from layer import *
from multi_layer_neuralnetwork import *
from heart_disease_health_indicators_BRFSS2015.dataset import make_data
import matplotlib.pyplot as plt


NeuralNetwork = Multi_Layer_NeuralNetwork(21, 8, 2)


(x_train, t_train), (x_test, t_test) = make_data()


epoch = 200
train_size = x_train.shape[0]
batch = 100
loop_per_epoch = int(train_size / batch)
leaening_rate = 0.005
train_loss_list = []
train_acc_list = []
test_acc_list = []


train_acc = NeuralNetwork.acc(x_train, t_train)
test_acc = NeuralNetwork.acc(x_test, t_test)
train_acc_list.append(train_acc)
test_acc_list.append(test_acc)


for i in range(epoch):
    for j in range(loop_per_epoch):
        batch_index = np.random.choice(train_size, batch, replace=True)
        x_batch_data = x_train[batch_index]
        t_batch_data = t_train[batch_index]
        grad = NeuralNetwork.grad_affine_params(x_batch_data, t_batch_data)
        affine_cnt = 0                                      # number of found affine layer
        for k in range(len(NeuralNetwork.layers)):          # search affine layer
            if isinstance(NeuralNetwork.layers[k], Affine): # find affine layer
                NeuralNetwork.layers[k].W -= leaening_rate * grad[affine_cnt][0]
                NeuralNetwork.layers[k].b -= leaening_rate * grad[affine_cnt][1]
                affine_cnt += 1
        loss = NeuralNetwork.loss(x_batch_data, t_batch_data)
        train_loss_list.append(loss)
    # 1 epoch done
    train_acc = NeuralNetwork.acc(x_train, t_train)
    test_acc = NeuralNetwork.acc(x_test, t_test)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    print("epoch={}, train_acc={}, test_acc={}".format(i+1, train_acc, test_acc))


def make_graph():
    fig = plt.figure()
    loss_graph = fig.add_subplot(1, 3, 1)
    train_graph = fig.add_subplot(1, 3, 2)
    test_graph = fig.add_subplot(1, 3, 3)
    loss_x = np.arange(epoch*loop_per_epoch)
    acc_x = np.arange(epoch+1)
    loss_graph.plot(loss_x, train_loss_list)
    loss_graph.set_title("train loss")
    loss_graph.set_xlabel("batch training times")
    loss_graph.set_ylabel("cross entropy error per 1 data")
    train_graph.plot(acc_x, train_acc_list)
    train_graph.set_title("train accuracy")
    train_graph.set_xlabel("epoch")
    train_graph.set_ylabel("accuracy")
    train_graph.set_ylim(0.4, 1)
    test_graph.plot(acc_x, test_acc_list)
    test_graph.set_title("test accuracy")
    test_graph.set_xlabel("epoch")
    test_graph.set_ylabel("accuracy")
    test_graph.set_ylim(0.4, 1)
    fig.tight_layout()
    plt.show()

make_graph()


# print(dataset.my_test)
# print(dataset.my_test.shape)
# print(NeuralNetwork.predict(dataset.my_test))
# print(train_acc_list)
# print(test_acc_list)
# print(train_loss_list)

