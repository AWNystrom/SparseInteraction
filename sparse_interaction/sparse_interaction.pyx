#!python
#cython: language_level=2, boundscheck=False, cdivision=True, overflowcheck=False, wraparound=False, initializedcheck=False

from __future__ import division
import numpy as np
cimport numpy as np
import cython
from sklearn.base import TransformerMixin, BaseEstimator
from scipy.sparse import hstack, coo_matrix
from itertools import combinations

from libc.math cimport pow

ctypedef np.uint64_t cUINT64
ctypedef np.double_t cDOUBLE

class SparseInteractionFeatures(TransformerMixin, BaseEstimator):
    """
    Compute interaction features on a sparse (CSR) matrix while keeping its format sparse.
    """
    
    def fit(self, X, Y=None, **fit_params):
        return self
    
    @cython.cdivision(True)
    def transform(self, X, **transform_params):
                                                        
        cdef cUINT64 a, b, x, y, row_num, interaction_col, n_cols_X, interaction_dim, nnz, num_interactions = 0
                 
        n_cols_X = X.shape[1]
        interaction_dim = <cUINT64>((n_cols_X**2-n_cols_X) / 2) #The size of the combination output space
        
        #http://math.stackexchange.com/questions/646117/how-to-find-a-function-mapping-matrix-indices
        #Thanks, Professor John Hughes of Brown University!
        
        #Get the number of nonzero interaction features
        for nnz in np.bincount(X.nonzero()[0]):
            num_interactions += <cUINT64>((pow(nnz,2)-nnz)/2)
        
        if num_interactions == 0:
            return X
            
        cdef np.ndarray[cUINT64, ndim=1, mode='c'] rows = np.ndarray(shape=num_interactions, dtype=np.uint64, order='C')
        cdef np.ndarray[cUINT64, ndim=1, mode='c'] cols = np.ndarray(shape=num_interactions, dtype=np.uint64, order='C')
        cdef np.ndarray[cDOUBLE, ndim=1, mode='c'] nz_data, data = np.ndarray(shape=num_interactions, dtype=np.double, order='C')
        cdef np.ndarray[cUINT64, ndim=1, mode='c'] nz_cols
        cdef cUINT64 interaction_index = 0, rownum = 0, num_nz_cols, col_a, col_b
        cdef cDOUBLE part1, part2, dat_a, dat_b
        
        for row in X:
            nz_cols = row.nonzero()[1].astype(np.uint64)
            nz_data = row[0, nz_cols].toarray().reshape((nz_cols.shape[0],)).astype(np.double)
            num_nz_cols = <cUINT64>nz_cols.shape[0]
            
            #I want these loops to be fast
            #for a, b in combinations(nz_cols, 2):
            for x in xrange(num_nz_cols):
                col_a = nz_cols[x]
                dat_a = nz_data[x]
                part1 = 2.*col_a*n_cols_X-pow(col_a,2)-3*col_a-2
                for y in range(x+1, num_nz_cols):
                    col_b = nz_cols[y]
                    dat_b = nz_data[y]
                    part2 = 2*col_b
                    interaction_col = <cUINT64>((part1+part2)/2)
                    rows[interaction_index] = rownum
                    cols[interaction_index] = interaction_col
                    data[interaction_index] = dat_a * dat_b
                    interaction_index += 1
            rownum += 1
                
        X_interaction = coo_matrix((data, (rows, cols)), 
                                    shape=(X.shape[0], interaction_dim), dtype=X.dtype).tocsr()
                                    
        return hstack((X, X_interaction)).tocsr()
    
    def __repr__(self):
        return "<SparseInteractionFeatures>"