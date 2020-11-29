import time
import numpy as np
import pandas as pd
from tqdm.auto import  tqdm
import numpy_indexed as npi

class reg():

    # Initializing the user-item rating matrix, no. of latent features
    def __init__(self, data):
        
        self.N, self.dim = data.iloc[:,:-1].shape
        self.data = data.copy().values
        
        self.beta = np.random.randn(5)

        self.beta_list = []
                

    # Initializing user-feature and item-feature matrix 
    def train(self,
              epoches, batch_size, lr_all=.005, reg_all=.02,
              lr_beta=None,reg_beta=None):
        
        self.batch_size = batch_size
        
        # learning rate
        self.lr_beta = lr_beta if lr_beta is not None else lr_all
        
        # regularization
        self.reg_beta = reg_beta if reg_beta is not None else reg_all    

        start = time.clock()
        alpha = 0.6
            
        for i in range(epoches):
            
            np.random.shuffle(self.data)
            self.data = self.data[:1000,:]
            self.N, self.dim = self.data[:,:-1].shape
            n = 1
            
            for ix in range(0, self.N, batch_size):
                
                x_arr = self.data[ix: ix+batch_size][:, :-1]
                Y = self.data[ix: ix+batch_size][:, -1].astype(int)
                
                #Gamma Decay      
        #        self.lr_beta = lr_beta * pow(n, -alpha)
                
                
                self.sgd(x_arr,Y)
                n += 1
                
                # store parameters
                self.beta_list.append(self.beta.copy())
    
    
    # Stochastic gradient descent to get optimized params
    def sgd(self, x_input, Y):

        batch = len(Y)
        
        X = x_input
      #  X = np.insert(x_input,0,1,axis=1)
        
        beta = self.beta.copy()
        
        update_beta = np.sum(X*np.expand_dims((Y - np.dot(X,beta)),1).repeat(X.shape[1],axis=1),axis=0)/batch
        
    #Normal case and update params
        self.beta += self.lr_beta * (update_beta - self.reg_beta * self.beta)
    

        return self
    

    def get_Y_hat(self, x_input):

        prediction = np.dot(x_input, self.beta)    
        
        return prediction