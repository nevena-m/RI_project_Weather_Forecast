import numpy as np
import random
import math

random.seed(a=None, version=2)

# Ulazne vrednosti i parametri
tests = 4; n = 2; m = 4; p = 1; maxiters = 100000
eta = 0.5; alpha = 0.9   
x = [[0, 0, 0], [1, 0, 0], [1, 1, 0],[1, 0, 1], [1, 1, 1]]
y = [[0, 0], [0, 0], [0, 1], [0, 1], [0, 0]]

h = [0.0 for i in range(m+1)]
dh = [0.0 for i in range(m+1)]
w_ = [[0.0 for i in range(m+1)] for j in range(n+1)]  
dw_ = [[0.0 for i in range(m+1)] for j in range(n+1)] 
w__ = [[0.0 for i in range(p+1)] for j in range(m+1)] 
dw__ = [[0.0 for i in range(p+1)] for j in range(m+1)]
o = [[0.0 for i in range(p+1)] for j in range(tests+1)]

# Inicijalizacija W'(ij)
for j in range(1, m+1):    
    for i in range(0, n+1):      
        w_[i][j] = random.random()  - 0.5
    
# Inicijalizacija W''(jk)
for k in range(1, p+1):   
    for j in range(0, m+1):
        w__[j][k] =  random.random() - 0.5
    
# Ucenje
for iters in range (0, maxiters): 
    # U svakoj iteraciji se bira proizvoljan od moguca 4 ulaza
    t = random.randrange(1,4+1)

    # Izracunavanje h1,..., hm
    h[0] = 1.0
    for j in range(1,m+1):  
        u = 0
        for i in range(0, n+1): 
            u += x[t][i] * w_[i][j]        
        h[j] = 1.0 / (1.0 + math.exp(-u))

    # Izracunavanje o1,...,op
    for k in range(1, p+1):    
        u = 0
        for j in range(0,m+1):
            u += h[j] * w__[j][k]
        o[t][k] = 1.0 / (1.0 + math.exp(-u))  
    
    # Izracunavanje deltaH(j)
    for j in range(1,m+1):   
        dh[j] = 0.0
        for k in range(1,p+1):
            dh[j] += w__[j][k] * (y[t][k] - o[t][k]) * o[t][k] * (1.0 - o[t][k])
    
    #Azuriranje W'(ij) i deltaW'(ij)
    for j in range(1,m+1):    
        for i in range(0,n+1): 
            dw_[i][j] = eta * x[t][i] * h[j] * (1 - h[j]) * dh[j] \
            + alpha * dw_[i][j] 
            w_[i][j] += dw_[i][j]
        
    #Azuriranje W''(jk) i deltaW''(jk)
    for k in range(1,p+1):   
        for j in range(0,m+1):
            dw__[j][k] = eta * h[j] * (y[t][k] - o[t][k]) \
            * o[t][k] * (1.0 - o[t][k]) \
            + alpha * dw__[j][k]        
            w__[j][k] += dw__[j][k]
        
    


# Testiranje
for i in range(1,5):
    print(x[i][1], x[i][2], end='')
    print(" ", y[i][0], o[i][0], y[i][1], o[i][1])
    
  

