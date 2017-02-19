#!python
#cython: language_level=2, boundscheck=False, cdivision=True, overflowcheck=False, wraparound=False, initializedcheck=False

from __future__ import division
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
cimport numpy as np


ctypedef np.float64_t DATA_T
ctypedef np.int64_t INDEX_T

__all__ = ['DensePolynomialFeatures']

class DensePolynomialFeatures(BaseEstimator, TransformerMixin):
  def __init__(self, degree=2):
    self.degree = degree
      
  def fit(self, X, Y=None, **fit_params):
    return self
    
  def transform(self, X, **transform_params):
    if self.degree == 2:
      return self.transform2(X)
    elif self.degree == 3:
      return self.transform3(X)
    else:
      raise Exception("Invalid degree %s" % (self.degree,))
  
  def transform2(self, X_):
    
    cdef np.ndarray[DATA_T, ndim=2] X = X_
    cdef INDEX_T degree = <INDEX_T>(self.degree)
    cdef INDEX_T N = X.shape[0]
    cdef INDEX_T D = X.shape[1]
    cdef INDEX_T i, j, row, poly_index
    cdef poly_D = D + <INDEX_T>((D**2+D)/2)
    cdef np.ndarray[DATA_T, ndim=2] poly_X = np.ndarray(shape=(N, poly_D), dtype=np.float64, order='C')
    cdef DATA_T data1
    
    row = 0
    while row < N:
      poly_index = 0
      i = 0
      while i < D:
        data1 = X[row, i]
        poly_X[row, poly_index] = data1
        poly_index += 1
        j = 0
        while j <= i:
          poly_X[row, poly_index] = data1 * X[row, j]
          poly_index += 1
          j += 1
        i += 1
      row += 1
      
    return poly_X
  
  def transform3(self, X_):
    cdef np.ndarray[DATA_T, ndim=2] X = X_
    cdef INDEX_T degree = <INDEX_T>(self.degree)
    cdef INDEX_T N = X.shape[0]
    cdef INDEX_T D = X.shape[1]
    cdef INDEX_T i, j, k, row, poly_index
    cdef poly_D = D + <INDEX_T>((D**2+D)/2) + <INDEX_T>((D**3+3*D**2+2*D)/6)
    cdef np.ndarray[DATA_T, ndim=2] poly_X = np.ndarray(shape=(N, poly_D), dtype=np.float64, order='C')
    cdef DATA_T data1, data2
    
    row = 0
    while row < N:
      poly_index = 0
      i = 0
      while i < D:
        data1 = X[row, i]
        poly_X[row, poly_index] = data1
        poly_index += 1
        j = 0
        while j <= i:
          data2 = X[row, j]
          poly_X[row, poly_index] = data1 * data2
          poly_index += 1
          k = 0
          while k <= j:
            poly_X[row, poly_index] = data1 * data2 * X[row, k]
            poly_index += 1
            k += 1
          j += 1
        i += 1
      row += 1
      
    return poly_X