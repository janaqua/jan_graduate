from OptimizationMethods import Batch_SGD
import time
import pickle
import numpy as np
import pandas as pd
from tqdm import  tqdm

for j in tqdm([1,2,5,10,20,30,50]):  #determine the nums of epoch to use
    
    
    batch_size = 500 
    
    # lr -> learning rate gamma
    # reg -> regularization
    
    lr_all = 1.6
    reg_all = 0.001
    
    #set learning rate and regularization for params individual
    lr_pu, lr_t, lr_beta = 2.1, 2.8, np.array([1.11, 1.052, 1.05])  
    reg_pu, reg_t, reg_beta = 0., 0., 0.
    
    
    epoches = j
    
    
    #create null variate for storing data    
    times = []
    rmse_p = []
    rmse_t = []
    rmse_beta = []
 
    #e_xxxx means thats the error_data 
    
    e_beta1 = []
    e_beta2 = []
    e_beta3 = []
    e_cdf = []
    e_threshold = []
    e_x = []
    e_p =[]
    e_prediction = []
    e_trues = [] 
    e_before_r = []
    
    for i in tqdm([0], ascii=True): #list裡面放要使用的資料及參數的號碼

        #loading data & get params        
        
        data = pd.read_csv('simulation_data_fix_param\data_%d.csv'%i)
        params = pickle.load(open('simulation_data_fix_param\parameters_%d.pkl'%i, 'rb'))
        P = params['P']
        t = params['t']
        beta = params['beta']

        #call out optimization and training data to fit params 
        
        model = Batch_SGD.ord_reg(data)
        start = time.clock()
        model.train(epoches, batch_size,
                    lr_all, reg_all,
                    lr_pu, lr_t, lr_beta,  
                    reg_pu, reg_t, reg_beta)
        times.append(time.clock() - start)

        # get rmse
        
        rmse_p.append(np.sqrt(np.mean((model.P - P)**2)))
        rmse_t.append(np.sqrt(np.mean((model.t - t)**2)))
        rmse_beta.append(np.sqrt(np.mean((model.beta - beta)**2)))
    
    
        # get error_data
    
        e_beta1.append(model.error_beta1)
        e_beta2.append(model.error_beta2)
        e_beta3.append(model.error_beta3)
        e_cdf.append(model.error_cdf)
        e_threshold.append(model.error_thresholds)
        e_x.append(model.error_x)
        e_p.append(model.error_p)
        e_prediction.append(model.error_prediction)
        e_trues.append(model.error_trues)
        e_before_r.append(model.get_before_error_r)
    
    #output rmse data    
    
    df = pd.DataFrame(index=['Batch SGD'], columns=['w', 't', 'beta', 'overall', 'training time'])
    df.loc['Batch SGD',:] = [np.mean(rmse_p),
                             np.mean(rmse_t),
                             np.mean(rmse_beta),
                             np.mean([np.mean(rmse_p),
                                      np.mean(rmse_t),
                                      np.mean(rmse_beta)]),
                             np.mean(times)]
    df.to_csv('ToConcatenate_fix_param\Batch_SGD_df%d.csv'%j)

    colname = [i for i in [0]]
    df_e = pd.DataFrame(columns = ['e_beta1','e_beta2','e_beta3','e_cdf',
                                 'e_threshold','e_x_input','e_p','e_prediction','e_trues','e_before_r'],
                        index = colname)
    
    df_e.loc[:,'e_beta1'] = e_beta1
    df_e.loc[:,'e_beta2'] = e_beta2
    df_e.loc[:,'e_beta3'] = e_beta3
    df_e.loc[:,'e_cdf'] = e_cdf
    df_e.loc[:,'e_threshold'] = e_threshold
    df_e.loc[:,'e_x_input'] = e_x
    df_e.loc[:,'e_p'] = e_p
    df_e.loc[:,'e_prediction'] = e_prediction
    df_e.loc[:,'e_trues'] = e_trues
    df_e.loc[:,'e_before_r'] = e_before_r

    if not ((df_e.apply(lambda x: len(x[0])) == 0).all()):
        df_e.to_csv('ToConcatenate_fix_param\ASGDandSGD_with_GammaDecay_df_error%d%d.csv'%j)






