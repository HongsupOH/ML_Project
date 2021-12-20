from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def gen_data(n):
    data = []
    x = np.linspace(-10,10,n)
    y = np.linspace(-10,10,n)
    
    x = np.kron(x, np.ones(n))
    y = np.array(y.tolist()*n)

    z = np.sin(np.log(x**2 + y**2))
    
    data = np.hstack((x.reshape(-1,1),\
                      y.reshape(-1,1),\
                      z.reshape(-1,1)))
    
    return data
