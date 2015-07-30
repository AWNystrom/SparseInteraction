import pyximport; pyximport.install()
import sparse_interaction
from scipy.sparse import csr_matrix, rand
from time import time
from sklearn.preprocessing import PolynomialFeatures
from code import interact
from numpy import  array_equal
interaction = sparse_interaction.SparseInteractionFeatures()

X = rand(10, 10000, 0.2).tocsr()#csr_matrix([[1,1,0,0],[0,0,1,1],[1,0,1,0],[0,1,0,1], [1,1,1,0], [0,1,1,0]])#
raw_input('made matrix')
#X = csr_matrix([[1,1,0,0],[0,0,1,1],[1,0,1,0],[0,1,0,1], [1,1,1,0], [0,1,1,0]])##rand(1000, 4, 0.5).tocsr()#

theirs = PolynomialFeatures(interaction_only=True)
t0 = time()
B = interaction.fit_transform(X)

t1 = time()
raw_input('transformed matrix')
print 'Mine:', t1-t0, B.shape

t0 = time()
C = theirs.fit_transform(X.toarray())
t1 = time()
print 'Theirs:', t1-t0, C.shape

#Get rid of the dummy feature sklearn adds
C = C[:,1:]

print B.toarray()
print C

print (B==C).sum() == B.shape[0]*B.shape[1] and B.shape==C.shape
