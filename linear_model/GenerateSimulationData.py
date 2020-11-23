import os
import numpy as np
import pandas as pd
import pickle

N = 50000 #the num of observation
dim = 5 #W

# specify the file path
try :
    os.mkdir('simulation_data')
except FileExistsError:
    pass

    
np.random.seed(2020)

# simulate_data
X = np.random.normal(size=(N, dim),loc=5,scale=1) 
beta = np.random.randn(dim)

error = np.random.randn(N)

Y = np.dot(X,beta)+error

#store data and output
data = pd.DataFrame(X)
data.loc[:, 'Y'] = Y

params = {'beta': beta,'sigma_square': 1}

pickle.dump(params, open('simulation_data\parameters_0.pkl', 'wb'))
data.to_csv('simulation_data\data_0.csv', index=False)
