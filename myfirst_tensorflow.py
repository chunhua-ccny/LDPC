import tensorflow as tf
import numpy as np

NumVector=10000
VecSize=1
NumHidden1=120
TrainRate=0.5
NUM_EPOCHS=3000
MINI_BATCH_SIZE=100

train_size = int(NumVector*TrainRate)*VecSize


all_x=np.random.normal(0, 3, (1,NumVector*VecSize)).T
all_y=np.zeros((NumVector*VecSize,1),dtype=np.float32)
for i in xrange(NumVector*VecSize):
  if(all_x[i]>0):
    all_y[i]=1
	
trainx = all_x[:train_size]
validx = all_x[train_size:]
trainy = all_y[:train_size]
validy = all_y[train_size:]
print 'sizes=',trainx.shape,validx.shape,trainy.shape,validy.shape
X = tf.placeholder(tf.float32, [None, 1], name="X")
Y = tf.placeholder(tf.float32, [None, 1], name="Y")


#W1=tf.Variable(tf.random_normal([VecSize,NumHidden1]))	
#b1=tf.Variable(tf.random_normal([VecSize,NumHidden1]))
#W2=tf.Variable(tf.random_normal([NumHidden1,VecSize]))	
#b2=tf.Variable(tf.random_normal([NumHidden1,VecSize]))

def init_weights(shape, init_method='xavier', xavier_params = (None, None)):
    if init_method == 'zeros':
        return tf.Variable(tf.zeros(shape, dtype=tf.float32))
    elif init_method == 'uniform':
        return tf.Variable(tf.random_normal(shape, stddev=0.01, dtype=tf.float32))
    else: #xavier
        (fan_in, fan_out) = xavier_params
        low = -4*np.sqrt(6.0/(fan_in + fan_out)) # {sigmoid:4, tanh:1} 
        high = 4*np.sqrt(6.0/(fan_in + fan_out))
        return tf.Variable(tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32))

def model(X, num_hidden=10):    
    w_h = init_weights([1, num_hidden], 'xavier', xavier_params=(1, num_hidden))
    b_h = init_weights([1, num_hidden], 'zeros')
    h = tf.nn.sigmoid(tf.matmul(X, w_h) + b_h)
    
    w_o = init_weights([num_hidden, 1], 'xavier', xavier_params=(num_hidden, 1))
    b_o = init_weights([1, 1], 'zeros')
    return tf.matmul(h, w_o) + b_o

#def model(X):
#  h1 = tf.nn.sigmoid(tf.matmul(X, W1) + b1)
#  h2 = tf.nn.sigmoid(tf.matmul(h1, W2) + b2)
#  return h2

yhat = model(X,NumHidden1)
train_op = tf.train.AdamOptimizer().minimize(tf.nn.l2_loss(yhat - Y))
sess = tf.Session()
sess.run(tf.initialize_all_variables())
errors=[]
for i in range(NUM_EPOCHS):
    for start, end in zip(range(0, len(trainx), MINI_BATCH_SIZE), range(MINI_BATCH_SIZE, len(trainx), MINI_BATCH_SIZE)):
        #print 'sess shape=',trainx[start:end].shape,trainy[start:end].shape
        sess.run(train_op, feed_dict={X: trainx[start:end], Y: trainy[start:end]})
    mse = sess.run(tf.nn.l2_loss(yhat - validy),  feed_dict={X:validx})
    #for j in range(train_size/VecSize):
	#  sess.run(train_op, feed_dict={X: (trainx[j*VecSize:(j+1)*VecSize]).reshape(-1,VecSize), Y: (trainy[j*VecSize:(j+1)*VecSize]).reshape(-1,VecSize)})
    #for start, end in zip(range(0, len(trainx), MINI_BATCH_SIZE), range(MINI_BATCH_SIZE, len(trainx), MINI_BATCH_SIZE)):
    #    sess.run(train_op, feed_dict={X: (trainx[start*VecSize:end*VecSize]).reshape(-1,VecSize), Y: (trainy[start*VecSize:end*VecSize]).reshape(-1,VecSize)})
    #mse = sess.run(tf.nn.l2_loss(yhat - validy),  feed_dict={X:validx})
    #if i%100==0: print 'yhat=', yhat, 'y=', validy
    errors.append(mse)
    if i%100 == 0: print "epoch %d, validation MSE %g" % (i, mse)
  
