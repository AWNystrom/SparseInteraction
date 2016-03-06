from scipy.sparse import random
from sklearn.preprocessing import PolynomialFeatures
from sparse_polynomial_features import SparsePolynomialFeatures
from time import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import vstack

"""
b: blue
g: green
r: red
c: cyan
m: magenta
y: yellow
k: black
w: white
"""

colors = ['b', 'r', 'm', 'y', 'm', 'c', 'g', 'k']

def plot_it(degree):
  Ds = np.arange(1, 6, 1)
  densities = np.arange(.1, 1.1, .1)#[0.0.1, .2, .3, .4, .5]
  times_sparse = [[0. for d in Ds] for d in densities]
  times_dense = [0. for d in Ds]
  iters = 5

  for iter in xrange(iters):
    for col, D in enumerate(Ds):
        for j, density in enumerate(densities):
          X = random(100000, D, density).tocsr()
          X_d = X.toarray()
          
          for name, l, trans in (('sparse', times_sparse, SparsePolynomialFeatures), ('dense', times_dense, PolynomialFeatures)):

            
            if name == 'sparse':
              t = trans(degree)
              a = time()
              t.fit_transform(X)
              b = time()
              l[j][col] += b-a
            else:
              t = trans(degree, include_bias=False)
              a = time()
              t.fit_transform(X_d)
              b = time()
              l[col] += b-a
          
  times_sparse = np.array(times_sparse) / iters
  times_dense = np.array(times_dense) / iters
  
  plt.hold = True
  #Plot sparse
  for density, times, c in zip(densities, times_sparse, colors):
    plt.plot(Ds, times, '%s:' % (c,), label='Sparse d=%s' % density)
  
  plt.plot(Ds, times_dense, 'k', label='scikit-learn')
  plt.xlabel('Dimensionality of Feature Matrix')
  plt.ylabel('Time to Compute Polynomial Features (seconds)')
  plt.title('Speed of scikit-learn vs Sparse Method (degree %s)' % (degree,))
  plt.legend(loc=2)
  plt.savefig('D_vs_time.pdf')
  

if __name__ == '__main__':
  plot_it(3)