import pandas as pd
import numpy as np
import tensorflow as tf
import threading

print("reading data")

data = np.fromfile('common-crawl/data/sorted',
        dtype=[('i', '<i4'), ('j', '<i4'), ('f', '<f4')],
        sep='')

data_len = data.shape[0]

batch_pos = 1

def get_batch(batch_size):
  global batch_pos
  # the remainder will not be missed
  if batch_pos + batch_size >= data_len:
    np.random.shuffle(data)
    batch_pos = 1
  batch = data[batch_pos : batch_pos + batch_size]
  batch_pos += batch_size

  return (batch['i'].astype(int) - 1, batch['j'].astype(int) - 1, batch['f'])

num_words = 500000
dimension = 250
batch_size = 500000

W     = tf.Variable(tf.random_normal([num_words, dimension], stddev=1/dimension))
W_hat = tf.Variable(tf.random_normal([num_words, dimension], stddev=1/dimension))
b     = tf.Variable(tf.random_normal([num_words], stddev=1/dimension))
b_hat = tf.Variable(tf.random_normal([num_words], stddev=1/dimension))

# TODO is it faster to send a single matrix?
batch_is = tf.placeholder(tf.int32, [batch_size])
batch_js = tf.placeholder(tf.int32, [batch_size])
batch_xs = tf.placeholder(tf.float32, [batch_size])

W_subset = tf.gather(W, batch_is)
W_hat_subset = tf.gather(W_hat, batch_js)
b_subset = tf.gather(b, batch_is)
b_hat_subset = tf.gather(b_hat, batch_js)

xmax = 100
alpha = 0.75

# TODO is it faster to precompute f for each entry?
f = tf.minimum(tf.pow(batch_xs / xmax, alpha), 1.0)

loss = tf.reduce_sum(
        tf.mul(f,
          tf.square(
              # (row i of W_subset) dot (row i of W_hat_subset)
              tf.einsum('ij,ji->i', W_subset, tf.transpose(W_hat_subset)) 
                + b_subset
                + b_hat_subset
                - tf.log(batch_xs))))

train_step = tf.train.AdamOptimizer(0.005).minimize(loss)

init = tf.global_variables_initializer()

print("initializing")

saver = tf.train.Saver([W, W_hat, b, b_hat])

sess = tf.Session()
sess.run(init)

print("training")

try:
  for i in range(3403 * 100):
    next_is, next_js, next_xs = get_batch(batch_size)
    sess.run(train_step,feed_dict={batch_is:next_is, batch_js:next_js, batch_xs:next_xs})
    if i % 100 == 0:
      print('iter {}, pass {}%, batch loss {}'.format(i, 100*batch_pos/data_len,
          sess.run(loss, feed_dict={batch_is:next_is, batch_js:next_js, batch_xs:next_xs})))
      saver.save(sess, 'data/ngrams-model', global_step=i)
except KeyboardInterrupt:
  pass
