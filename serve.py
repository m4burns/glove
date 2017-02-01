import nearest as n
import pca as p
import numpy as np
from bottle import run, post, request, response, get, route, static_file
import json
import math

def getk(request):
  try:
    return int(request.query.k)
  except ValueError:
    return 10

@route('/nn/<word>', method = 'GET')
def getnns(word):
  response.content_type = 'application/json'
  return json.dumps(n.nn(n.vec(word), k=getk(request)))

@route('/pcann/<word>', method = 'GET')
def getpcanns(word):
  response.content_type = 'application/json'
  nnv = n.nnv(n.vec(word), k = getk(request))
  words = n.words(nnv.indices)
  vecs = n.index_w(nnv.indices)
  pca, eigval, eigvec = p.PCA(vecs)
  pca -= pca.mean(axis=0)
  pca /= np.abs(pca).max(axis=0)
  res = [ { 'w': word if word == word else '', 'p': pca_vec.tolist(), 'd': float(val) } for word, pca_vec, val in zip(words, pca, nnv.values) ]
  return json.dumps(res)

@route('/')
def server_static():
  return static_file('/index.html', root='./static')

@route('/<filepath:path>')
def server_static(filepath):
  return static_file(filepath, root='./static')

run(host='0.0.0.0', port=8080, debug=True)
