from __future__ import division
import numpy as np
cimport numpy as np

ctypedef np.float64_t DTYPE_t
ctypedef np.int_t INDEX_t

cdef second_deg_poly_float(np.ndarray[INDEX_t, ndim=2] rows, 
                           np.ndarray[INDEX_t, ndim=2] cols, 
                           np.ndarray[DTYPE_t, ndim=2] data):

  