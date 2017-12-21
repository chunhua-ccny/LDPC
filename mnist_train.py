# coding=utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

"""
    mnist 数据集了解
    每个图片大小：28 x 28
"""
# 训练集
train_img = mnist.train.images
print('train_img.shape: '),
print train_img.shape
train_label = mnist.train.labels
print('train_label.shape: '),
print train_label.shape
# 测试集
test_img = mnist.test.images
test_label = mnist.test.labels
# 验证集
validate_img = mnist.validation.images
validate_label = mnist.validation.labels

#def gen_image(arr):
#    two_d = (np.reshape(arr, (28, 28)) * 255).astype(np.uint8)
#    plt.imshow(two_d, interpolation='nearest')
#    return plt

# Get a batch of two random images and show in a pop-up window.
#batch_xs, batch_ys = mnist.test.next_batch(2)
#print 'test label 0=',mnist.test.labels[0]
#print 'test label 1=',mnist.test.labels[1]
#gen_image(batch_xs[0]).show()
#gen_image(batch_xs[1]).show()

print np.argmax(mnist.test.labels[0:100],axis=1).reshape(10,10)
Images = mnist.test.images[0:100] #get 100x784 array
Images = np.array(Images, dtype='float')
ImagShow=np.zeros((10*28,10*28),dtype=np.float32)
for i in range(10): #0--9
  for j in range(10): #0--9
    imageId=i*10+j
    imageTemp=Images[imageId].reshape((28,28))
    ImagShow[i*28:(i+1)*28,j*28:(j+1)*28]=imageTemp;	
plt.imshow(ImagShow, cmap='gray')
plt.show()


# 构造计算图
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])
# loss function　代价函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# 用于评估准确率
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)
# 训练
# 用于记录每次训练后loss的值
loss_val = []
# 用于记录每次训练后模型在测试集上的准确率的值
acc = []
for idx in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    loss_tmp = sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys})
    loss_val.append(loss_tmp)
    acc_tmp = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    acc.append(acc_tmp)

# 绘图
plt.figure()
plt.plot(loss_val)
plt.xlabel('number of iteration')
plt.ylabel('loss value')

plt.figure()
plt.plot(acc)
plt.xlabel('number of iteration')
plt.ylabel('accuracy value')
plt.show()