from scipy.sparse import random, vstack
from sparse_polynomial_features import SparsePolynomialFeatures
from dense_polynomial_features import DensePolynomialFeatures as PolynomialFeatures
from time import time
import numpy as np
import matplotlib.pyplot as plt
from code import interact
import cPickle
from sys import argv
import seaborn as sns

np.random.seed(42)

"""
We have variables D, d, N, t

Vary D, d, N while keeping the others constant and plot it vs time.

Constant slices: D=500, N=1000, d=0.5

Do this with k = 2, then all over with k=3

This will yield 6 plots

"""


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

poly_order = 2

filename = 'order_%s_simple_data.pickle' % (poly_order,)
colors = ['b', 'r', 'm', 'y', 'm', 'c', 'g', 'k']
iters = 20

density_steps = 10.
ds = np.arange(density_steps + 1) / density_steps

#if poly_order == 2:
#  Ds = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
#  Ns = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

#  D_slice = int(np.median(Ds))
#  d_slice = 0.2
#  N_slice = int(np.median(Ns))
  
#elif poly_order == 3:

if poly_order == 3:
  Ns = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
  max_D = 250.
  Ds = map(int, np.arange(0, max_D+max_D/10, max_D/10))
  Ds[0] = 1
  N_slice = int(np.median(Ds))
elif poly_order == 2:
  Ns = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
  max_D = 1000.
  Ds = map(int, np.arange(0, max_D+max_D/10, max_D/10))
  Ds[0] = 1
  N_slice = int(max(Ds))

D_slice = int(np.median(Ds))
d_slice = 0.2

#else:
#  assert(False)

# Key to dep_to_times is always (N, D, d)
var_sets = [
             {
               'title': 'Density (d) vs Time (D=%s, N=%s)' % (D_slice, N_slice,),
               'xlabel': 'Matrix Density',
               'ylabel': 'Mean Time in Seconds (%s trials)' % (iters,),
               'ds': ds,
               'Ds': [D_slice],
               'Ns': [N_slice],
               'variation_ind': 2,
               'dep_to_times_sparse': {(N_slice, D_slice, dep):[] for dep in ds},
               'dep_to_times_dense': {(N_slice, D_slice, dep):[] for dep in ds},
             },
             
             {
               'title': 'Dimensionality (D) vs Time (d=%s, N=%s)' % (d_slice, N_slice,),
               'xlabel': 'Matrix Dimensionality',
               'ylabel': 'Mean Time in Seconds (%s trials)' % (iters,),
               'ds': [d_slice],
               'Ds': Ds,
               'Ns': [N_slice],
               'variation_ind': 1,
               'dep_to_times_sparse': {(N_slice, dep, d_slice):[] for dep in Ds},
               'dep_to_times_dense': {(N_slice, dep, d_slice):[] for dep in Ds},
             },
             
             {
               'title': 'Instance Count (N) vs Time (D=%s, d=%s)' % (D_slice, d_slice,),
               'xlabel': 'Matrix Row Count',
               'ylabel': 'Mean Time in Seconds (%s trials)' % (iters,),
               'ds': [d_slice],
               'Ds': [D_slice],
               'Ns': Ns,
               'variation_ind': 0,
               'dep_to_times_sparse': {(dep, D_slice, d_slice):[] for dep in Ns},
               'dep_to_times_dense': {(dep, D_slice, d_slice):[] for dep in Ns},
             },
           ]

def fill_times():
  for iter in range(iters):
    for var_set in var_sets:
      for N in var_set['Ns']:
        for D in var_set['Ds']:
          for d in var_set['ds']:
            print(iter, var_set['title'], (N, D, d))
            
            X_sparse = vstack((random(1, D, d) for i in range(N))).tocsr()
            X_dense = X_sparse.toarray()
          
            a = time()
            SparsePolynomialFeatures(poly_order,
                                     interaction_only=False).fit_transform(X_sparse)
            t = time() - a
            var_set['dep_to_times_sparse'][(N, D, d)].append(t)
            
            a = time()
            PolynomialFeatures(poly_order).fit_transform(X_dense)
            t = time() - a
            var_set['dep_to_times_dense'][(N, D, d)].append(t)
            print(t)

def make_plots(param_sets):
  fig, axes = plt.subplots(1, 3, sharex=False, sharey=False, figsize=(15,5))
  assert(len(param_sets) == 3)
  
  for axis, param_set in zip(axes, param_sets):
  
    i = param_set['variation_ind']
    
    #interact(local=locals())
    
    #ax2.fill_between(densities, low_sparse, high_sparse, color="#3F5D7D")
    
    X, Y = zip(*[(x[i], np.mean(ys)) for x, ys in param_set['dep_to_times_sparse'].items()])
    sort_inds = np.argsort(X)
    X = np.array(X)[sort_inds]
    Y = np.array(Y)[sort_inds]
    axis.plot(X, Y, 'b', label='Sparse Algorithm')
    
    X, Y = zip(*[(x[i], np.mean(ys)) for x, ys in param_set['dep_to_times_dense'].items()])
    sort_inds = np.argsort(X)
    X = np.array(X)[sort_inds]
    Y = np.array(Y)[sort_inds]
    axis.plot(X, Y, 'r', label='Dense Algorithm')
    
    #plt.title(param_set['title'])
    axis.set_xlabel(param_set['xlabel'])
    axis.set_ylabel(param_set['ylabel'])
  plt.legend()
  plt.savefig(filename.replace(' ', '_') + '.png')
#  plt.show()

def plot_both_orders(second_orders, third_orders):
  second_orders.sort(key=lambda item: item['title'])
  third_orders.sort(key=lambda item: item['title'])
  
  fig, axes = plt.subplots(2, 3, sharex=False, sharey=False, figsize=(15,10))
  row = 0
  for row, col, param_set in zip([0,0,0,1,1,1], [0,1,2,0,1,2], second_orders + third_orders):
    axis = axes[row, col]
    i = param_set['variation_ind']
    
    X, Y = zip(*[(x[i], np.mean(ys)) for x, ys in param_set['dep_to_times_sparse'].items()])
    sort_inds = np.argsort(X)
    X = np.array(X)[sort_inds]
    Y = np.array(Y)[sort_inds]
    axis.plot(X, Y, 'b--.', label='Sparse Algorithm')
    
    X, Y = zip(*[(x[i], np.mean(ys)) for x, ys in param_set['dep_to_times_dense'].items()])
    sort_inds = np.argsort(X)
    X = np.array(X)[sort_inds]
    Y = np.array(Y)[sort_inds]
    axis.plot(X, Y, 'r--.', label='Dense Algorithm')
    
    axis.set_title(param_set['title'])
    axis.set_xlabel(param_set['xlabel'])
    axis.set_ylabel(param_set['ylabel'])
    plt.legend()
  plt.legend()
  plt.savefig(filename.replace(' ', '_') + '.png')
  plt.show()
  
if __name__ == '__main__':
#  fill_times()
#  cPickle.dump(var_sets, open(filename, 'w'), 2)
#  param_sets = cPickle.load(open(filename, 'r'))
#  make_plots(param_sets)

  second_orders = cPickle.load(open('order_%s_simple_data.pickle' % (2,), 'r'))
  third_orders = cPickle.load(open('order_%s_simple_data.pickle' % (3,), 'r'))
  
  plot_both_orders(second_orders, third_orders)