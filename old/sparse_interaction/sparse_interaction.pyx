#!python
#cython: language_level=2, boundscheck=False, cdivision=True, overflowcheck=False, wraparound=False, initializedcheck=False

from __future__ import division
import numpy as np
cimport numpy as np
import cython
from sklearn.base import TransformerMixin, BaseEstimator
from scipy.sparse import coo_matrix

from libc.math cimport pow

ctypedef np.uint64_t INDEXDTYPE

class SparsePolynomialFeatures(TransformerMixin, BaseEstimator):
    """
    Compute interaction features on a sparse (CSR) matrix while keeping its format sparse.
    """
    
    def __init__(self, interaction_only=False):
        self.interaction_only = interaction_only
    
    def fit(self, X, Y=None, **fit_params):
        return self
    
    @cython.cdivision(True)
    def transform(self, X, int interaction_only=False, int combine=True, **transform_params):
                                                        
        cdef INDEXDTYPE a, b, x, y, row_num, interaction_col, n_cols_X, output_dim, nnz, num_interactions = 0, min_col_count
        cdef int combo_j_offset, combo_i_offset
        
        n_cols_X = X.shape[1]
        if interaction_only == 1:
            combo_j_offset = 1
            combo_i_offset = -1
            output_dim = <INDEXDTYPE>((n_cols_X**2-n_cols_X) / 2) #upper triangular
            min_col_count = 2
            for nnz in np.bincount(X.nonzero()[0]):
                num_interactions += <INDEXDTYPE>((pow(nnz,2)-nnz)/2)
        else:
            combo_j_offset = 0
            combo_i_offset = 0
            min_col_count = 1
            output_dim = <INDEXDTYPE>((n_cols_X**2+n_cols_X) / 2) #diagonal counted
            for nnz in np.bincount(X.nonzero()[0]):
                num_interactions += <INDEXDTYPE>((pow(nnz,2)+nnz)/2)
        
        if combine:
            num_interactions += X.nnz
            output_dim += X.shape[1]
            
        if num_interactions == 0:
            return X
        
        cdef np.ndarray[INDEXDTYPE, ndim=1, mode='c'] rows = np.ndarray(shape=num_interactions, dtype=np.uint64, order='C')
        cdef np.ndarray[INDEXDTYPE, ndim=1, mode='c'] cols = np.ndarray(shape=num_interactions, dtype=np.uint64, order='C')
        cdef np.ndarray[double, ndim=1, mode='c'] nz_data, data = np.ndarray(shape=num_interactions, dtype=np.double, order='C')
        cdef np.ndarray[INDEXDTYPE, ndim=1, mode='c'] nz_cols
        cdef INDEXDTYPE interaction_index = 0, rownum = 0, num_nz_cols, col_a, col_b
        cdef double part1, part2, dat_a, dat_b
        
        for row in X:
            #print 'rownum', rownum
            nz_cols = row.nonzero()[1].astype(np.uint64)
            nz_data = row[0, nz_cols].toarray().reshape((nz_cols.shape[0],)).astype(np.double)
            num_nz_cols = <INDEXDTYPE>nz_cols.shape[0]
            
            if combine:
                for col, dat in zip(nz_cols, nz_data):                
                    rows[interaction_index] = rownum
                    cols[interaction_index] = col
                    data[interaction_index] = dat
                    interaction_index += 1
            
            #No interactions if not at least 2 nz elements, no poly if not at least 1
            if num_nz_cols >= min_col_count:
                for x in xrange(num_nz_cols+combo_i_offset):
                    col_a = nz_cols[x]
                    dat_a = nz_data[x]
                    part1 = 2.*col_a*n_cols_X-pow(col_a,2)-3*col_a-2
                        
                    for y in range(x+combo_j_offset, num_nz_cols):
                        col_b = nz_cols[y]
                        dat_b = nz_data[y]
                        part2 = 2*col_b
                        
                        if interaction_only == 1:
                            interaction_col = <INDEXDTYPE>((part1+part2)/2)
                        else:
                            interaction_col = <INDEXDTYPE>(1 + col_a + (part1+part2)/2)
                        
                        if combine:
                            interaction_col += n_cols_X
                            
                        rows[interaction_index] = rownum
                        cols[interaction_index] = interaction_col
                        data[interaction_index] = dat_a * dat_b
                        interaction_index += 1
            rownum += 1
        
        X_interaction = coo_matrix((data, (rows, cols)), 
                                    shape=(X.shape[0], output_dim), dtype=X.dtype).tocsr()
        
        return X_interaction
    
    def __repr__(self):
        return "<SparsePolynomialFeatures>"
