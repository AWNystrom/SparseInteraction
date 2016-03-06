from scipy.sparse import csr_matrix
from numpy import array, empty, int32, zeros

def second_deg(X):
  data, indices, indptr = X.data, X.indices, X.indptr
  
  #Count the number of nonzero items in each row
  poly_nz_count = 0
  for i in xrange(indptr.shape[0]-1):
    D = indptr[i+1] - indptr[i]
    poly_nz_count += D + (D**2+D)/2
  
  #Make the arrays that will form the new CSR matrix
  new_data = zeros(shape=(poly_nz_count,), dtype=X.dtype, order='C')
  new_indices = zeros(shape=(poly_nz_count,), dtype='int32', order='C')
  new_indptr = zeros(shape=(X.indptr.shape[0],), dtype='int32', order='C')
  
  print 'poly_nz_count', poly_nz_count
  
  new_indptr[0] = 0
  
  ind = 0
  
  original_D = X.shape[1]
  
  #Calculate the poly features
  for i in xrange(indptr.shape[0]-1):
    start = indptr[i]
    stop = indptr[i+1]
    
    cols = indices[start:stop]
    
    num_cols = 0
    
    for k1 in xrange(start, stop):
      new_data[ind] = data[k1]
      new_indices[ind] = indices[k1]
      print ind
      ind += 1
      num_cols += 1
    
    for k2 in xrange(start, stop):
      col2 = indices[k2]
      data2 = data[k2]
      
      #Add the poly features
      for k1 in xrange(start, k2+1):
        col1 = indices[k1]
        data1 = data[k1]
        poly_col = (2*col1+col2**2+col2+1)/2 + original_D
        poly_data = data1*data2
        
        #Now put everything in its right place
        print ind
        new_data[ind] = poly_data
        new_indices[ind] = poly_col
        ind += 1
        num_cols += 1
    
    new_indptr[i+1] = new_indptr[i] + num_cols
  
  print
  print 'data', new_data
  print
  print 'indices', new_indices
  print
  print 'indptr', new_indptr
  print
  
  A = csr_matrix([])
  A.data = new_data
  A.indices = new_indices
  A.indptr = new_indptr
  A._shape = (X.shape[0], X.shape[1] + (X.shape[1]**2 + X.shape[1]) / 2)
  return A
  

def create_csr_from_array(A):
  from scipy.sparse import csr_matrix
  data = []
  indices = []
  indptr = [0]
  
  nnz = 0
  for i, row in enumerate(A):
    for c, d in enumerate(row):
      if d != 0:
        indices.append(c)
        data.append(d)
        nnz += 1
    indptr.append(nnz)
  
  return csr_matrix((data, indices, indptr), shape=A.shape)
  
if __name__ == '__main__':
  from scipy.sparse import random
  from sklearn.preprocessing import PolynomialFeatures
  import numpy as np
  from time import time
  
  X = random(2, 3, .7).tocsr()
  
  print 'actual nnz', csr_matrix(PolynomialFeatures(2, include_bias=False).fit_transform(X.toarray())).nnz
  
  print X.toarray()
  
  print second_deg(X).toarray()
  
  #a=time()
  xp = csr_matrix(PolynomialFeatures(2, include_bias=False).fit_transform(X.toarray()))
  
  print xp.indptr
  print xp.indices
  print xp.shape
  print xp.data
  #print time()-a
  
  #print sum(sum(np.ceil(clf.fit_transform(X.toarray()))))
  
  """
  d^k < 1/a
  k < -ln(a)/ln(d)
  
  """