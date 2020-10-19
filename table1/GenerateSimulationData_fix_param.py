import os
import numpy as np
import pandas as pd
import pickle
from tqdm.auto import  tqdm

N = 50000 #the num of observation
dim = 2 #W
r = 5 # rating


# specify the file path
try :
    os.mkdir('simulation_data_fix_param')
except FileExistsError:
    pass





for i in tqdm([1]):
    
    np.random.seed(2020)
    
    # simulate_data
    X = np.random.normal(size=(N, dim)) 
    P = np.random.randn(dim)
    t = np.random.randn(1)
    beta = np.random.randn(r - 2)
    
    thresold = np.append(t, np.exp(beta)).cumsum()
    
    y = np.dot(X, P)

    # compute distribution functions - cdf
    F = 1 /(1 + np.exp(y[:, np.newaxis] - thresold)) 
    F = np.append(np.zeros(F.shape[0])[:, np.newaxis], np.append(F, np.ones(F.shape[0])[:, np.newaxis], axis=1), axis=1)
    
    
    mass = np.diff(F, axis=1) #pdf
    score = np.apply_along_axis(func1d=lambda x: np.asscalar(np.random.choice(list(range(r)), size=1, p=x)),
                                arr=mass, axis=1) + 1 #sample the rating
                                
    #store data and output
    data = pd.DataFrame(X)
    data.loc[:, 'score'] = score
    
    params = {'P': P, 
              't': t,
              'beta': beta}
    
    pickle.dump(params, open('simulation_data_fix_param\parameters_%d.pkl'%i, 'wb'))
    data.to_csv('simulation_data_fix_param\data_%d.csv' % i, index=False)
