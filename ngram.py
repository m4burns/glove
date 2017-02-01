import pandas as pd
import numpy as np
import tensorflow as tf
import threading

print("reading data")

data = np.fromfile('common-crawl/data/sorted',
            dtype=[('i', '<i4'), ('j', '<i4'), ('f', '<f4')],
            sep='')
#data = pd.read_csv('X', delimiter=' ', dtype=float).as_matrix()
data_len = data.shape[0]

print(str(data_len) + " nonzeros")

batch_pos = 1

def get_batch(batch_size):
  global batch_pos
  #indices = (np.random.random((batch_size))*data_len).astype(int)
  #batch = data[np.random.choice(data_len,size=batch_size,replace=False)]
  if batch_pos + batch_size >= data_len:
    np.random.shuffle(data)
    batch_pos = 1
  batch = data[batch_pos:batch_pos+batch_size]
  batch_pos += batch_size

  # return (batch[:,0].astype(int) - 1, batch[:,1].astype(int) - 1, batch[:,2])
  return (batch['i'].astype(int) - 1, batch['j'].astype(int) - 1, batch['f'])

num_words = 500000
dimension = 250

W     = tf.Variable(tf.random_normal([num_words, dimension],stddev=1/dimension))
W_hat = tf.Variable(tf.random_normal([num_words, dimension],stddev=1/dimension))
b     = tf.Variable(tf.random_normal([num_words],stddev=1/dimension))
b_hat = tf.Variable(tf.random_normal([num_words],stddev=1/dimension))

batch_size = 500000

# train_q = tf.FIFOQueue(2, (tf.int32, tf.int32, tf.float32), [[batch_size],[batch_size],[batch_size]])

batch_is = tf.placeholder(tf.int32, [batch_size])
batch_js = tf.placeholder(tf.int32, [batch_size])
batch_xs = tf.placeholder(tf.float32, [batch_size])

# enqueue_op = train_q.enqueue([async_batch_is, async_batch_js, async_batch_xs])

# batch_is, batch_js, batch_xs = train_q.dequeue()

W_subset = tf.gather(W, batch_is)
W_hat_subset = tf.gather(W_hat, batch_js)
b_subset = tf.gather(b, batch_is)
b_hat_subset = tf.gather(b_hat, batch_js)

# TODO f(X_ij)

xmax = 100
alpha = 0.75
f = tf.minimum(tf.pow(batch_xs / xmax, alpha), 1.0)

loss = tf.reduce_sum(
        tf.mul(f,
          tf.square(
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

coord = tf.train.Coordinator()

# def train_q_filler():
#   print("started q filler")
#   while not coord.should_stop():
#     next_is, next_js, next_xs = get_batch(batch_size)
#     sess.run(enqueue_op, feed_dict={async_batch_is: next_is, async_batch_js: next_js, async_batch_xs: next_xs})
# 
# train_q_filler_thread = threading.Thread(target=train_q_filler)
# train_q_filler_thread.daemon = True
# train_q_filler_thread.start()

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
