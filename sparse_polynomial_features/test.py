import pyximport; pyximport.install()
from sparse_polynomial_features import SparsePolynomialFeatures
from sklearn.preprocessing import PolynomialFeatures
import unittest
from scipy.sparse import random
from code import interact
import cPickle
from numpy import array
from numpy.random import choice
from scipy.sparse import csr_matrix

primes = cPickle.load(open('primes.pickle'))

"""
def test_this(rows, cols, density, interaction_only, degree):
  print(rows, cols, density, interaction_only, degree)
  X_sparse = random(rows, cols, density).tocsr()
  X_dense = X_sparse.toarray()
  
  poly_d = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False).fit_transform(X_dense)
  
  try:
    poly_s = SparsePolynomialFeatures(degree=degree, interaction_only=interaction_only).fit_transform(X_sparse).toarray()
  except:
    interact(local=locals())

  f1 = get_val_dist(poly_d)
  f2 = get_val_dist(poly_s)
  
  all_vals = set()
  all_vals.update(f1.keys())
  all_vals.update(f2.keys())
  
  for v in all_vals:
    assert f1[v] == f2[v]

def get_val_dist(X):
  f = {}
  for v in X.flatten():
    f[v] = f.get(v, 0) + 1
  return f

for interaction_only in [True]:#[True, False]:
  for degree in [2, 3]:
    for density in [0., 0.5, 1.]:
      for rows in [1, 10]:
        for cols in [1, 10]:
          test_this(rows, cols, density, interaction_only, degree)
"""

def via_primes(N, D, degree, interaction_only):

  X_dense = array(choice(primes, N*D, replace=False), dtype=float).reshape((N, D))
  X_sparse = csr_matrix(X_dense)
  poly_d = PolynomialFeatures(degree=degree, interaction_only=interaction_only,
                              include_bias=False).fit_transform(X_dense)
  poly_s = SparsePolynomialFeatures(degree=degree, 
                                    interaction_only=interaction_only).fit_transform(X_sparse).toarray()

  #Figure out the mapping on the first one, then ensure that it holds for all the others.
  #row_d and row_s should always agree in the same spot. 
  sparse_inds_vals = sorted(enumerate(poly_s[0, :]), key=lambda item: item[1])
  dense_inds_vals = sorted(enumerate(poly_d[0, :]), key=lambda item: item[1])
  sparse_inds = zip(*sparse_inds_vals)[0]
  dense_inds = zip(*dense_inds_vals)[0]
  s_to_d_ind = dict(zip(sparse_inds, dense_inds))
  
  for row in xrange(1, N):
    row_d = poly_d[row, :].flatten()
    row_s = poly_s[row, :].flatten()
    assert all(row_s[j] == row_d[s_to_d_ind[j]] for j in range(D))
    print(len(row_s.flatten()))

if __name__ == '__main__':
  via_primes(10, 50, 2, False)