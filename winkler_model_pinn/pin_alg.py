import numpy as np
import dem_hyperelasticity.hertzian_contact.config as cf

def set_pinball(dom,idx):
    Na = cf.Na
    Nr = cf.Nr
    n = len(idx)
    dim = 2
    X = np.empty((n-1,4,dim))
    c = np.empty((n-1,dim))
    R = np.empty((n,1))
    for i in (idx):
        if i<dom.shape[0]:
            X[i,0] = dom[i]
            X[i,1] = dom[i+1]
            X[i,2] = dom[i-Na]
            X[i,3] = dom[i-Na+1]
            c[i] = (X[i,0]+X[i,1]+X[i,2]+X[i,3])/4
            A = 0.5*abs((X[i,0,0]*X[i,1,1]-X[i,1,0]*X[i,0,1])+(X[i,1,0]*X[i,2,1]-X[i,2,0]*X[i,1,1])+(X[i,2,0]*X[i,3,1]-X[i,3,0]*X[i,2,1])+(X[i,3,0]*X[i,0,1]-X[i,0,0]*X[i,3,1]))
            R = np.sqrt(A/np.pi)

    return X, c, R

def distance(c1,c2):
    x=0
    _, dim = c1.shape
    if dim != c2.shape[1]:
        raise Exception("dimension is not equal")
    
    for i in range(dim):
        x += (c1[0,i]-c2[0,i])**2
    dis = np.sqrt(x)
    return dis

def normal(c2,c1):
    n = (c2-c1)/distance(c1,c2)
    return n

def gap(c1,c2,R1,R2):
    n = normal(c2,c1)
    b = n@(c2-c1).T
    c = distance(c1,c2)**2-(R1+R2)**2
    g = -b+np.sqrt(b**2-c)
    return g

def center_2d(b_1,b_2):
    c_1 = np.array([[(b_1[0,0]+b_1[1,0]+b_1[2,0]+b_1[3,0])/4,(b_1[0,1]+b_1[1,1]+b_1[2,1]+b_1[3,1])/4]])
    c_2 = np.array([[(b_2[0,0]+b_2[1,0]+b_2[2,0]+b_2[3,0])/4,(b_2[0,1]+b_2[1,1]+b_2[2,1]+b_2[3,1])/4]])
    return c_1, c_2