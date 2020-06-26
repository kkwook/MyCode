from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images.shape
len(train_labels)
test_images.shape
len(test_images)

from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation="relu", input_shape = (28 * 28,)))
network.add(layers.Dense(10, activation = "softmax"))
network.compile(optimizer='rmsprop',
                loss = "categorical_crossentropy",
                metrics=["accuracy"])

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
train_images.shape
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255
test_images.shape
from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
train_labels.shape
test_labels  = to_categorical(test_labels)

network.fit(train_images, train_labels, epochs = 5, batch_size = 128)
network.summary()

test_loss, test_acc = network.evaluate(test_images, test_labels)

from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
digit = train_images[4]
import matplotlib.pyplot as plt
plt.imshow(digit, cmap = plt.cm.binary)
plt.show()

''' 텐서 연산 '''
import keras
keras.layers.Dense(512, activation = "relu")
output = relu(dot(W, input) + b)

''' 지역 최솟값에 머물지 않고 전역 최솟값까지 가게 만들기 '''
past_velocity = 0
momentum = 0.1
while loss > 0.01:
    w, loss, gradient = get_current_parameters()
    velocity = momentum * past_velocity - learning_rate * gradient
    w = w + momentum * velocity - learning_rate * gradient
    past_velocity = velocity
    update_parameter(w)








