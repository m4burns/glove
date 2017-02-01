import pandas as pd
import numpy as np
import tensorflow as tf
import sys
import fileinput

dict = pd.read_csv('common-crawl/data/dict_r', delimiter=' ')

idx2word = dict.set_index('idx').to_dict()['word']
word2idx = dict.set_index('word').to_dict()['idx']

# TODO duplication
num_words = 500000
dimension = 250

W_out_actual = None
W_normalized_actual = None

g_comp = tf.Graph()

# build W + W_hat and a normalized copy from checkpoint
with g_comp.as_default():
  W     = tf.Variable(tf.random_normal([num_words, dimension]))
  W_hat = tf.Variable(tf.random_normal([num_words, dimension]))
  b     = tf.Variable(tf.random_normal([num_words]))
  b_hat = tf.Variable(tf.random_normal([num_words]))
  
  W_out = W + W_hat
  
  W_lengths = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(W_out), 1)),[num_words,1])
  W_normalized = tf.div(W_out, W_lengths)
  
  saver = tf.train.Saver([W, W_hat, b, b_hat])
  
  with tf.Session(config=tf.ConfigProto(device_count = {'GPU': 0})) as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint('data/'))
    W_out_actual = sess.run(W_out)
    W_normalized_actual = sess.run(W_normalized)

tf.reset_default_graph()

W_out_comp = tf.constant(W_out_actual)
W_normalized_comp = tf.constant(W_normalized_actual)

nearest_arg = tf.placeholder(tf.float32, [dimension])
nearest_arg_length = tf.sqrt(tf.reduce_sum(tf.square(nearest_arg)))
nearest_arg_normalized = nearest_arg / nearest_arg_length

k_arg = tf.placeholder(tf.int32, [])

dist_all = tf.reduce_sum(W_normalized_comp * nearest_arg_normalized, 1)
dist_top_k = tf.nn.top_k(dist_all, k_arg)

sess = tf.Session(config=tf.ConfigProto(device_count = {'GPU': 0}))
sess.run(tf.global_variables_initializer())

def vec(word):
  return W_out_actual[word2idx[word]-1]

def nnv(vec, k=10, dist_op=dist_top_k):
  return sess.run(dist_op, feed_dict={nearest_arg: vec, k_arg: k})

def nn(vec, k=10, dist_op=dist_top_k):
  return words(nnv(vec,k,dist_op).indices)

def index_w(indices):
  return W_out_actual[indices]

def words(indices):
  return [idx2word[i+1] for i in indices]

def vecs(words):
  return index_w([word2idx[word]-1 for word in words])
