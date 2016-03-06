# SparseInteraction
This is a polynomial feature generator that operates directly on a csr_matrix and leverages data sparsity to achive time and space complexity of O(d^kD^k) where d is the density of the vector, D is the dimensionality, and k is the polynomial degree. The code currently supports 2nd and 3rd degree expansions.
