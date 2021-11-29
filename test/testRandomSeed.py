import numpy as np

seq_seed = [1001, 2001, 3001]

for seed in seq_seed:
    np.random.seed(seed)
    print(np.random.poisson(1, 5))
    print(np.random.poisson(1,5))

np.random.seed(1001)
print(np.random.poisson(1,5))
print(np.random.poisson(1,5))

np.random.seed(3001)
print(np.random.poisson(1,5))
print(np.random.poisson(1,5))