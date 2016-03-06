#!python
#cython: language_level=2, boundscheck=False, cdivision=True, overflowcheck=False, wraparound=False, initializedcheck=False

from __future__ import division
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
cimport numpy as np


ctypedef np.float64_t DATA_T
ctypedef np.int32_t INDEX_T

__all__ = ['SparsePolynomialFeatures']
    
def sec_deg_poly_feats(X):
    cdef np.ndarray[DATA_T, ndim=1] data = X.data
    cdef np.ndarray[INDEX_T, ndim=1] indices = X.indices, indptr = X.indptr
  
    #Count the number of nonzero items in each row
    cdef INDEX_T poly_nz_count = 0, i = 0, D
  
    while i < indptr.shape[0]-1:
      D = indptr[i+1] - indptr[i]
      poly_nz_count += D + <INDEX_T>((D**2+D)/2)
      i += 1
  
    #Make the arrays that will form the new CSR matrix
    cdef np.ndarray[DATA_T, ndim=1] new_data = np.ndarray(shape=poly_nz_count, dtype=np.float64, order='C')
    cdef np.ndarray[INDEX_T, ndim=1] new_indices = np.ndarray(shape=poly_nz_count, dtype=np.int32, order='C')
    cdef np.ndarray[INDEX_T, ndim=1] new_indptr = np.ndarray(shape=indptr.shape[0], dtype=np.int32, order='C')
  
    new_indptr[0] = 0
    cdef INDEX_T ind = 0, start, stop, num_cols, k1, k2, col2, original_D = X.shape[1]
    cdef DATA_T data2
    i = 0
    
    #Calculate the poly features
    while i < indptr.shape[0]-1:
      start = indptr[i]
      stop = indptr[i+1]    
      num_cols = 0
    
      k1 = start
      while k1 < stop:
        new_data[ind] = data[k1]
        new_indices[ind] = indices[k1]
        ind += 1
        num_cols += 1
        k1 += 1
    
      k2 = start
      while k2 < stop:
        col2 = indices[k2]
        data2 = data[k2]
      
        #Add the poly features
        k1 = start
        while k1 < k2+1:
          #Now put everything in its right place
          new_data[ind] = data[k1]*data2
          new_indices[ind] = (2*indices[k1]+col2**2+col2+1)/2 + original_D
          ind += 1
          num_cols += 1
          k1 += 1
        k2 += 1
    
      new_indptr[i+1] = new_indptr[i] + num_cols
      i += 1
    
    A = csr_matrix([])
    A.data = new_data
    A.indices = new_indices
    A.indptr = new_indptr
    A._shape = (X.shape[0], X.shape[1] + <INDEX_T>((X.shape[1]**2 + X.shape[1]) / 2))
    return A

def sec_deg_inter_feats(X):
    cdef np.ndarray[DATA_T, ndim=1] data = X.data
    cdef np.ndarray[INDEX_T, ndim=1] indices = X.indices, indptr = X.indptr
  
    #Count the number of nonzero items in each row
    cdef INDEX_T poly_nz_count = 0, i = 0, D
  
    while i < indptr.shape[0]-1:
      D = indptr[i+1] - indptr[i]
      poly_nz_count += D + <INDEX_T>((D**2-D)/2)
      i += 1
  
    #Make the arrays that will form the new CSR matrix
    cdef np.ndarray[DATA_T, ndim=1] new_data = np.ndarray(shape=poly_nz_count, dtype=np.float64, order='C')
    cdef np.ndarray[INDEX_T, ndim=1] new_indices = np.ndarray(shape=poly_nz_count, dtype=np.int32, order='C')
    cdef np.ndarray[INDEX_T, ndim=1] new_indptr = np.ndarray(shape=indptr.shape[0], dtype=np.int32, order='C')
  
    new_indptr[0] = 0
    cdef INDEX_T ind = 0, start, stop, num_cols, k1, k2, col2, original_D = X.shape[1]
    cdef DATA_T data2
    i = 0
    
    #Calculate the poly features
    while i < indptr.shape[0]-1:
      start = indptr[i]
      stop = indptr[i+1]    
      num_cols = 0
    
      k1 = start
      while k1 < stop:
        new_data[ind] = data[k1]
        new_indices[ind] = indices[k1]
        ind += 1
        num_cols += 1
        k1 += 1
    
      k2 = start
      while k2 < stop:
        col2 = indices[k2]
        data2 = data[k2]
      
        #Add the poly features
        k1 = start
        while k1 < k2:
          #Now put everything in its right place
          new_data[ind] = data[k1]*data2
          new_indices[ind] = (2*indices[k1]+col2**2+col2+1)/2 + original_D
          ind += 1
          num_cols += 1
          k1 += 1
        k2 += 1
    
      new_indptr[i+1] = new_indptr[i] + num_cols
      i += 1
    
    A = csr_matrix([])
    A.data = new_data
    A.indices = new_indices
    A.indptr = new_indptr
    A._shape = (X.shape[0], X.shape[1] + <INDEX_T>((X.shape[1]**2 - X.shape[1]) / 2))
    return A


def third_deg_poly_feats(X):
    cdef np.ndarray[DATA_T, ndim=1] data = X.data
    cdef np.ndarray[INDEX_T, ndim=1] indices = X.indices, indptr = X.indptr
  
    #Count the number of nonzero items in each row
    cdef INDEX_T poly_nz_count = 0, i = 0, D
  
    while i < indptr.shape[0]-1:
      D = indptr[i+1] - indptr[i]
      poly_nz_count += D + <INDEX_T>((D**2+D)/2) + <INDEX_T>((D**3+3*D**2+2*D)/6)
      i += 1
  
    #Make the arrays that will form the new CSR matrix
    cdef np.ndarray[DATA_T, ndim=1] new_data = np.ndarray(shape=poly_nz_count, dtype=np.float64, order='C')
    cdef np.ndarray[INDEX_T, ndim=1] new_indices = np.ndarray(shape=poly_nz_count, dtype=np.int32, order='C')
    cdef np.ndarray[INDEX_T, ndim=1] new_indptr = np.ndarray(shape=indptr.shape[0], dtype=np.int32, order='C')
  
    new_indptr[0] = 0
    cdef INDEX_T ind = 0, start, stop, num_cols, k1, k2, k3, col2, col3, original_D = X.shape[1], second_order_D
    cdef DATA_T data2, data3
    i = 0
    
    second_order_D = <INDEX_T>((original_D**2 + original_D) / 2)
    
    #Calculate the poly features
    while i < indptr.shape[0]-1:
      start = indptr[i]
      stop = indptr[i+1]    
      num_cols = 0
    
      #first degree
      k1 = start
      while k1 < stop:
        new_data[ind] = data[k1]
        new_indices[ind] = indices[k1]
        ind += 1
        num_cols += 1
        k1 += 1
      
      #Second degree
      k2 = start
      while k2 < stop:
        col2 = indices[k2]
        data2 = data[k2]
      
        #Add the poly features
        k1 = start
        while k1 < k2+1:
          #Now put everything in its right place
          new_data[ind] = data[k1]*data2
          new_indices[ind] = (2*indices[k1]+col2**2+col2+1)/2 + original_D
          ind += 1
          num_cols += 1
          k1 += 1
        k2 += 1
      
      #Third degree
      k3 = start
      while k3 < stop:
        col3 = indices[k3]
        data3 = data[k3]
        k2 = start
        while k2 < k3+1:
          col2 = indices[k2]
          data2 = data[k2]
          k1 = start
          while k1 < k2+1:
            #Now put everything in its right place
            new_data[ind] = data[k1]*data2*data3
            new_indices[ind] = (col3**3 + 3*col3**2 + 2*col3)/6 + (col2**2+col2)/2 + indices[k1] + original_D + second_order_D
            ind += 1
            num_cols += 1
            k1 += 1
          k2 += 1
        k3 += 1
    
      new_indptr[i+1] = new_indptr[i] + num_cols
      i += 1
    
    A = csr_matrix([])
    A.data = new_data
    A.indices = new_indices
    A.indptr = new_indptr
    D = X.shape[1]
    A._shape = (X.shape[0], D + <INDEX_T>((D**2 + D) / 2) + <INDEX_T>((D**3 + 3*D**2 + 2*D)/6))
    return A

def third_deg_inter_feats(X):
    cdef np.ndarray[DATA_T, ndim=1] data = X.data
    cdef np.ndarray[INDEX_T, ndim=1] indices = X.indices, indptr = X.indptr
  
    #Count the number of nonzero items in each row
    cdef INDEX_T poly_nz_count = 0, i = 0, D
  
    while i < indptr.shape[0]-1:
      D = indptr[i+1] - indptr[i]
      poly_nz_count += D + <INDEX_T>((D**2-D)/2) + <INDEX_T>((D**3-3*D**2+2*D)/6)
      i += 1
  
    #Make the arrays that will form the new CSR matrix
    cdef np.ndarray[DATA_T, ndim=1] new_data = np.ndarray(shape=poly_nz_count, dtype=np.float64, order='C')
    cdef np.ndarray[INDEX_T, ndim=1] new_indices = np.ndarray(shape=poly_nz_count, dtype=np.int32, order='C')
    cdef np.ndarray[INDEX_T, ndim=1] new_indptr = np.ndarray(shape=indptr.shape[0], dtype=np.int32, order='C')
  
    new_indptr[0] = 0
    cdef INDEX_T ind = 0, start, stop, num_cols, k1, k2, k3, col2, col3, original_D = X.shape[1], second_order_D
    cdef DATA_T data2, data3
    i = 0
    
    second_order_D = <INDEX_T>((original_D**2 - original_D) / 2)
    
    #Calculate the poly features
    while i < indptr.shape[0]-1:
      start = indptr[i]
      stop = indptr[i+1]    
      num_cols = 0
    
      #first degree
      k1 = start
      while k1 < stop:
        new_data[ind] = data[k1]
        new_indices[ind] = indices[k1]
        ind += 1
        num_cols += 1
        k1 += 1
      
      #Second degree
      k2 = start
      while k2 < stop:
        col2 = indices[k2]
        data2 = data[k2]
      
        #Add the poly features
        k1 = start
        while k1 < k2:
          #Now put everything in its right place
          new_data[ind] = data[k1]*data2
          new_indices[ind] = (2*indices[k1]+col2**2-col2)/2 + original_D
          ind += 1
          num_cols += 1
          k1 += 1
        k2 += 1
      
      #Third degree
      k3 = start
      while k3 < stop:
        col3 = indices[k3]
        data3 = data[k3]
        k2 = start
        while k2 < k3:
          col2 = indices[k2]
          data2 = data[k2]
          k1 = start
          while k1 < k2:
            #Now put everything in its right place
            new_data[ind] = data[k1]*data2*data3
            new_indices[ind] = indices[k1] + (col2**2-col2)/2 + ((col3-2)**3 + 3*(col3-2)**2 + 2*(col3-2))/6 + original_D + second_order_D
            ind += 1
            num_cols += 1
            k1 += 1
          k2 += 1
        k3 += 1
    
      new_indptr[i+1] = new_indptr[i] + num_cols
      i += 1
    
    A = csr_matrix([])
    A.data = new_data
    A.indices = new_indices
    A.indptr = new_indptr
    D = X.shape[1]
    A._shape = (X.shape[0], D + <INDEX_T>((D**2 - D) / 2) + <INDEX_T>((D**3 - 3*D**2 + 2*D)/6))
    return A

params_to_funct = {(2, False): sec_deg_poly_feats, 
                   (2, True): sec_deg_inter_feats,
                   (3, False): third_deg_poly_feats,
                   (3, True): third_deg_inter_feats
                  }

class SparsePolynomialFeatures(BaseEstimator, TransformerMixin):
  def __init__(self, degree=2, interaction_only=False):
    self.degree = degree
    self.interaction_only = interaction_only
    self.__p = (self.degree, self.interaction_only)
    if self.__p not in params_to_funct:
      raise Warning("SparsePolynomialFeatures does not support these options.")
      
  def fit(self, X, Y=None, **fit_params):
    return self
  
  def transform(self, X, **transform_params):
    return params_to_funct[self.__p](X)