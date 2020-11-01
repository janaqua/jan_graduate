from OptimizationMethods import Batch_SGD_GammaDecay

import time
import pickle
import numpy as np
import pandas as pd
from tqdm import  tqdm


beta_bar_hat = []
sigma_bar_hat = []
for i in tqdm(range(1000)):

    
    for j in [50]:  #determine the nums of epoch to use
        
        
        batch_size = 500 
        
        # lr -> learning rate gamma
        # reg -> regularization
        
        lr_all = 1.6
        reg_all = 0.001
        
        #set learning rate and regularization for params individual
        lr_sigma_square,lr_beta = 1.5,np.array([1.11, 1.052, 1.05,1.2,1.15,1.1])  
        
        reg_beta ,reg_sigma_square= 0.0, 0.0
            
        epoches = j
        
        
        #create null variate for storing data    
        times = []
        rmse_beta = []
        rmse_sigma_square = []
        
        rmse_avg_beta = []
        rmse_avg_sigma_square = []
        
    
        #loading data & get params        
        
        data = pd.read_csv('simulation_data\data_0.csv')
        
        params = pickle.load(open('simulation_data\parameters_0.pkl', 'rb'))
        beta = params['beta']
        sigma_square = params['sigma_square']
        
        #call out optimization and training data to fit params 
        
        model = Batch_SGD_GammaDecay.reg(data)
        start = time.clock()
        model.train(epoches, batch_size,
                    lr_all, reg_all,
                    lr_beta, lr_sigma_square, 
                    reg_beta, reg_sigma_square)
        
        times.append(time.clock() - start)
        
        beta_avg = np.mean(model.beta_list, axis=0)
        sigma_square_avg = np.mean(model.sigma_square_list)
        
        
        '''
        # get rmse
        rmse_beta.append(np.sqrt(np.mean((model.beta - beta)**2)))
        rmse_sigma_square.append(np.sqrt(np.mean((model.sigma_square - sigma_square)**2)))
    
        rmse_avg_beta.append(np.sqrt(np.mean((beta_avg - beta)**2)))
        rmse_avg_sigma_square.append(np.sqrt(np.mean((sigma_square_avg - sigma_square)**2)))
    
       #output rmse data    
     
        df = pd.DataFrame(index=['Batch SGD with gamma_n decayed', 'Batch ASGD with gamma_n decayed'],
                          columns=['beta','sigma_square', 'overall', 'training time'])
        df.loc['Batch SGD with gamma_n decayed',:] = [np.mean(rmse_beta),
                                                np.mean(rmse_sigma_square),
                                                np.mean([np.mean(rmse_beta),
                                                         np.mean(rmse_sigma_square)]),
                                                np.mean(times)]
        
        df.loc['Batch ASGD with gamma_n decayed',:] = [np.mean(rmse_avg_beta),
                                                np.mean(rmse_avg_sigma_square),
                                                np.mean([np.mean(rmse_avg_beta),
                                                         np.mean(rmse_avg_sigma_square)]),
                                                np.mean(times)]
        
        
        df.to_csv('ToConcatenate\Batch_ASGDandSGD_with_GammaDecay_df%d.csv'%j)
        '''
        beta_bar_hat.append(beta_avg)
        sigma_bar_hat.append(sigma_square_avg)


#%%
data1 = pd.DataFrame(beta_bar_hat)
data2 = pd.DataFrame(sigma_bar_hat)

data1.to_csv('1000_times_beta.csv',index =False)
data2.to_csv('1000_times_sigma.csv',index = False)