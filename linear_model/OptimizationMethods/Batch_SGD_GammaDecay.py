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
        
        self.beta = np.random.randn(6)
        self.sigma_square = np.random.uniform(0.5,5,1)
        self.beta_list = []
        self.sigma_square_list = []
                

    # Initializing user-feature and item-feature matrix 
    def train(self,
              epoches, batch_size, lr_all=.005, reg_all=.02,
              lr_beta=None, lr_sigma_square = None,  
              reg_beta=None, reg_sigma_square = None):
        
        self.batch_size = batch_size
        
        # learning rate
        self.lr_beta = lr_beta if lr_beta is not None else lr_all
        self.lr_sigma_square = lr_sigma_square if lr_sigma_square is not None else lr_all
        
        # regularization
        self.reg_beta = reg_beta if reg_beta is not None else reg_all
        self.reg_sigma_square = reg_sigma_square if reg_sigma_square is not None else reg_all
        

        start = time.clock()
        alpha = 0.5
        
        for i in range(epoches):
            
            np.random.shuffle(self.data)
            n = 1
            for ix in range(0, self.N, batch_size):
                
                x_arr = self.data[ix: ix+batch_size][:, :-1]
                Y = self.data[ix: ix+batch_size][:, -1].astype(int)
                
                #Gamma Decay      
                self.lr_beta = lr_beta * pow(n, -alpha)
                self.lr_sigma_square = lr_sigma_square * pow(n, -alpha)
                
                
                
                self.sgd(x_arr,Y)
                n += 1
                
                # store parameters
                self.beta_list.append(self.beta.copy())
                self.sigma_square_list.append(self.sigma_square.copy())

    
    # Stochastic gradient descent to get optimized params
    def sgd(self, x_input, Y):

        batch = len(Y)

        X = np.insert(x_input,0,1,axis=1)
        
        beta = self.beta.copy()
        sigma_square = self.sigma_square.copy()
        
        update_beta =  np.sum(X*np.expand_dims((Y - np.dot(X,beta)),1).repeat(X.shape[1],axis=1),axis=0)
        update_beta /= batch*sigma_square
        
        update_sigma_square = -(1/(2*sigma_square))+ np.sum((Y - np.dot(X,beta))**2)/(batch*(2*sigma_square**2)) 
        
    #Normal case and update params
        self.beta += self.lr_beta * (update_beta - self.reg_beta * self.beta)
        self.sigma_square += self.lr_sigma_square \
            *(update_sigma_square - self.reg_sigma_square*self.sigma_square)


        return self
    

    def get_Y_hat(self, x_input):

        prediction = self.beta[0]+np.dot(x_input, self.beta[1:])    
        
        return prediction
