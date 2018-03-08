from itertools import combinations, combinations_with_replacement

T2 = lambda n: (n * (n+1)) / 2
T3 = lambda n: (n * (n+1) * (n+2)) / 6

mappings = {'2i': lambda D, i, j: T2(D-1) - (T2(D-i-1) - (j-i-1)),
            '2p': lambda D, i, j: T2(D) - (T2(D-i) - (j-i)),
            '3i': lambda D, i, j, k: T3(D-2) - (T3(D-i-3) + T2(D-j-1) - (k-j-1)),
            '3p': lambda D, i, j, k: T3(D) - (T3(D-i-1) + T2(D-j) - (k-j))}

def DoIt(D, degree, kind):
  assert kind in 'ip'
  assert degree in (2, 3)
  code = '%s%s' % (degree, kind)
  mapping = mappings[code]
  inds = range(D)
  iter_funct = combinations if kind is 'i' else combinations_with_replacement
  
  expected = 0
  for combo in iter_funct(inds, degree):
    output_ind = mapping(D, *combo)
    assert expected == output_ind
    expected += 1
  print expected
  
def DoAll():
  for D in [5, 10, 20, 100]:
    DoIt(D, 2, 'i')
    DoIt(D, 2, 'p')
    DoIt(D, 3, 'i')
    DoIt(D, 3, 'p')

if __name__ == '__main__':
  DoAll()