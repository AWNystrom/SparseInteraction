from time import time
from scipy.sparse import rand
from sparse_interaction import SparsePolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from code import interact
from numpy import arange, median
from memory_profiler import profile
import StringIO

N = 1000
iters=1

sp = SparsePolynomialFeatures()
poly = PolynomialFeatures(include_bias=False)

dD_to_times = {}
other_d_to_times={}

min_D = 1
D_step = 100
max_D = 500+D_step


min_d = 0.2
d_step = .2
max_d = 1.

def speed_test():
    for i in xrange(iters):
        for d in arange(min_d, max_d, d_step):
            for D in arange(min_D, max_D, D_step):
                X = rand(N, D, d).tocsr()
                X_d = X.toarray()
        
                a = time()
                A = sp.transform(X)
                t = time()-a
                if d not in dD_to_times:
                    dD_to_times[d] = {}
            
                dD_to_times[d][D] = dD_to_times[d].get(D, 0) + t
                print i, d, D, t
            
                if d == max_d-d_step:
                    a = time()
                    B = poly.fit_transform(X_d)
                    t = time()-a
                    other_d_to_times[D] = other_d_to_times.get(D, 0) + t


    for d in sorted(dD_to_times.keys()):
        for D in dD_to_times[d].keys():
            dD_to_times[d][D] /= iters
        pairs = dD_to_times[d].items()
        pairs.sort()
        x, y = zip(*pairs)
        plt.plot(x, y, label='d=%s'%(d,))
    
    for k, val in other_d_to_times.iteritems():
        other_d_to_times[k] /= iters

    pairs = other_d_to_times.items()
    pairs.sort()
    x, y = zip(*pairs)
    plt.plot(x, y, label='scikit-learn', color='red', marker='*')

    plt.title('Times of Polynomial Feature Generation on Random Matrices With %s Rows' % (N,))
    plt.xlabel('Dimensionality')
    plt.ylabel('Time to Calculate (seconds)')
    plt.legend(loc='best')
    plt.xlim([0, max_D-D_step])
    plt.grid(True)
    plt.savefig('some_fig.svg')
    interact(local=locals())
    plt.show()

poly = PolynomialFeatures(include_bias=False)
sp = SparsePolynomialFeatures()

X_sparse = None
X_dense = None
def do_sklearn():
    global X_dense
    poly.fit_transform(X_dense)

def do_sparse_poly():
    global X_sparse
    sp.transform(X_sparse)

from re import compile
mem_re = compile('\d+\.\d+')
def memit(f):
    s = StringIO.StringIO()
    profile(f, s, precision=2)()
    s.seek(0)
    out = s.read()
    print out
    nums = mem_re.findall(out)
    print nums
    return float(nums[-1]) + float(nums[-2])
    
    
    
def ram_test():
    for d in arange(min_d, max_d, d_step):
        for D in arange(min_D, max_D, D_step):
            print d, D
            X = rand(N, D, d).tocsr()
            X_d = X.toarray()
            
            global X_sparse
            X_sparse = X
            global X_dense
            X_dense = X_d
            t = memit(do_sparse_poly)
            if d not in dD_to_times:
                dD_to_times[d] = {}
        
            dD_to_times[d][D] = dD_to_times[d].get(D, 0) + t
            print d, D, t
        
            if d == max_d-d_step:
               
                t = memit(do_sklearn)
                other_d_to_times[D] = other_d_to_times.get(D, 0) + t


for d in sorted(dD_to_times.keys()):
    for D in dD_to_times[d].keys():
        dD_to_times[d][D] /= iters
    pairs = dD_to_times[d].items()
    pairs.sort()
    x, y = zip(*pairs)
    plt.plot(x, y, label='d=%s'%(d,))
    
    for k, val in other_d_to_times.iteritems():
        other_d_to_times[k] /= iters

    pairs = other_d_to_times.items()
    pairs.sort()
    x, y = zip(*pairs)
    plt.plot(x, y, label='scikit-learn', color='red', marker='-')

    plt.title('Space of Polynomial Feature Generation on Random Matrices With %s Rows' % (N,))
    plt.xlabel('Dimensionality')
    plt.ylabel('RAM to Calculate (MB)')
    plt.legend(loc='best')
    plt.xlim([0, max_D-D_step])
    plt.grid(True)
    plt.xlim([0, 550])
    plt.ylim([0, 8])
    plt.savefig('some_fig.pdf')
    interact(local=locals())
    plt.show()

if __name__ == '__main__':
    speed_test()