from memory_profiler import profile
from sparse_interaction import SparsePolynomialFeatures
from sklearn.preprocessing import PolynomialFeatures
from cPickle import load
from scipy.sparse import csr_matrix
from sklearn.datasets import fetch_20newsgroups_vectorized
from time import time

#@profile
def sparse(X):
    si = SparsePolynomialFeatures()
    X_i = si.transform(X)
    
#@profile
def dense(X):
    poly = PolynomialFeatures(include_bias=False)
    X_i = poly.fit_transform(X)

if __name__ == '__main__':
    #X_filename = 'connect_4/X'
    #X = load(open(X_filename))
    #X_sparse = csr_matrix(X)
    X_sparse = fetch_20newsgroups_vectorized().data
    X = X_sparse.toarray()

    a=time(); dense(X); print time()-a
    #a=time(); sparse(X_sparse); print time()-a
