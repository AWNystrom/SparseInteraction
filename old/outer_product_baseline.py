from numpy import outer, triu_indices
from scipy.sparse import csr_matrix, coo_matrix, triu
from code import interact

def second_deg_poly_features(X):
  D = X.shape[1]
  D2 = (D**2+D)/2+D
  print D2
  X2 = coo_matrix(X.tocsr(), shape=(X.shape[0], D2)).tocsr()
  for i, row in enumerate(X):
    interact(local=locals())
    nzrows, nzcols = triu(outer(row.T,row)[0][0]).flatten()
    print r.shape
    X2[i, D:] = r
  return X2