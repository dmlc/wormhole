#!/usr/bin/python
"""
kmeans: 
"""
import os
import sys
import numpy as np
from scipy import sparse
# import rabit, the tracker script will setup the lib path correctly
# for normal run without tracker script, add following line
sys.path.append('../../../rabit/wrapper')
sys.path.append('../../dmlc-core/wrapper')
import rabit
import dmlc_core

class Model(object):
  def __init__(me, nbr_cluster, fdim):
    me.centroid = np.zeros((nbr_cluster, fdim))
    me.K = nbr_cluster
    me.F = fdim
  def InitCentroids(me, data):
    data.BeforeFirst()
    if not data.Next():
      print 'error empty data'
    spmat = data.ValueCSR()
    for i in range(me.K):
      rowid = np.random.randint(0, data.length - 1)
      me.centroid[i] += spmat.getrow(rowid)
    rabit.broadcast(me.centroid, 0)
  def normalize(me):
    norm = np.sqrt(me.centroid.sum(axis = 1))
    me.centroid  /=  norm.reshape((norm.size, 1))

def main():

  num_cluster = int(sys.argv[2])
  max_iter = int(sys.argv[3])
  rabit.init(sys.argv)
  world_size = rabit.get_world_size()
  rank = rabit.get_rank()
  data_iter = dmlc_core.RowBlockIter()
  data_iter.CreateFromUri(sys.argv[1], rank, world_size, 'libsvm')
  iter_cnt = 0
  fdim_array = np.array([0])
  fdim_array[0] = data_iter.NumCol() 
  #print fdim_array
  if iter_cnt == 0:
    fdim_array = rabit.allreduce(fdim_array, rabit.MAX)
    model = Model(num_cluster, int(fdim_array[0]))
    model.InitCentroids(data_iter)
    #model.normalize()
  num_feat = fdim_array[0]
  data_iter.setNumFeat(num_feat)
  for it in range(iter_cnt, max_iter):
    if rabit.get_rank() == 0:
      print 'iter = ', it

    temp = np.zeros((num_cluster, num_feat + 1), dtype=np.float32)

    def preparefun(temp):
      nbrline = 0
      data_iter.BeforeFirst()
      while data_iter.Next():
        spmat = data_iter.ValueCSR()
        nbrline += spmat.shape[0]
        num_row = spmat.shape[0]
        
        vnorm = np.sqrt(spmat.multiply(spmat).sum(axis = 1)) 
        dotp = spmat.dot(model.centroid.T)
        dist = dotp / vnorm
        max_id = np.argmax(dist, axis = 1)
        for k in range(num_cluster):
          temp[:,num_feat] += np.where(max_id == k)[0].shape[1]
        data_iter.CSRReduceSum(max_id, temp)
        #print 'processed %d lines = ' % (nbrline)
        
    rabit.allreduce(temp, rabit.SUM, preparefun)
    model.centroid = temp[:,0:num_feat]
      #print temp
    for k in range(num_cluster):
      assert(temp[k,num_feat] > 0)
      model.centroid[k,:] /= temp[k,num_feat]
      #print model.centroid
        #dist /= 
    #model.normalize()
  rabit.finalize()
main()