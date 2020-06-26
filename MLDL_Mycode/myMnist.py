from myAnn import TwoLayerNet
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

''' 미니 배치 다시 연습 '''
(x_train, t_train), (x_test, t_test) = load_mnist(
    normalize=True, one_hot_label=True)
print(x_train.shape) # 입력층
print(t_train.shape) # 출력층

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000 # 반복 횟수 설정
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

tarin_liss_list = []
tarin_acc_list = []
test_acc_list = []

# 1 에폭시 당 반복 수
iter_per_epoch = max(train_size / batch_size, 1)

a = np.array([1,2,3,4,5])
a
a.size
a = a.reshape(1, a.size)
a
a.shape



