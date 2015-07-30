import pyximport; pyximport.install()
import sparse_interaction
from scipy.sparse import csr_matrix, rand
from time import time
from sklearn.preprocessing import PolynomialFeatures
from code import interact

interaction = sparse_interaction.SparseInteractionFeatures()

X = rand(1000, 182, 0.2).tocsr()#csr_matrix([[1,1,0,0],[0,0,1,1],[1,0,1,0],[0,1,0,1], [1,1,1,0], [0,1,1,0]])#
raw_input('made matrix')
#X = csr_matrix([[1,1,0,0],[0,0,1,1],[1,0,1,0],[0,1,0,1], [1,1,1,0], [0,1,1,0]])##rand(1000, 4, 0.5).tocsr()#

theirs = PolynomialFeatures(interaction_only=True)
t0 = time()
B = interaction.fit_transform(X)

t1 = time()
raw_input('transformed matrix')
print 'Mine:', t1-t0, B.shape

#t0 = time()
#C = theirs.fit_transform(X.toarray())
#t1 = time()
#print 'Theirs:', t1-t0, C.shape

#print X.toarray()
print B.toarray()
