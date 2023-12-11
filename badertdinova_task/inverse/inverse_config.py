import numpy as np

# Initialize constant for our task
# Initialize constant for our task
Pk = 8.5e+6
m = 0.134
betta = 1e-10
H = 5
rc = 0.1
r0 = 100
Rk = 100
Q = 7.8/86400
t0 = 0
t_end = 5*86400
ro_oil0 = 800
S = 2*np.pi*H
device = 'cuda'

# initial guess
G = 1. # *1e+4
C_skv = 1. # *1e-6
k = 1. # *1e-14
mu_h = 1. # *25e-3