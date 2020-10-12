from OptimizationMethods import SGD
import time
import pickle
import numpy as np
import pandas as pd
from tqdm import  tqdm

for j in tqdm([1,2,5,10,20,30,50]):
    lr_all = 1.6
    reg_all = 0.001 
    lr_pu, lr_t, lr_beta = 0.18, 0.2, np.array([0.18, 0.02, 0.023])  
    reg_pu, reg_t, reg_beta = 0., 0., 0.
    epoches = j
    
    times = []
    rmse_p = []
    rmse_t = []
    rmse_beta = []
    
    e_beta1 = []
    e_beta2 = []
    e_beta3 = []
    e_bigger = []
    e_smaller = []
    e_cdf = []
    e_threshold = []
    e_x = []
    e_p =[]
    e_prediction = []
    e_trues = []
    
    for i in tqdm([0], ascii=True):
        
        data = pd.read_csv('simulation_data_fix_param\data_%d.csv'%i)
        params = pickle.load(open('simulation_data_fix_param\parameters_%d.pkl'%i, 'rb'))
        P = params['P']
        t = params['t']
        beta = params['beta']
        
        model = SGD.ord_reg(data)
        start = time.clock()
        model.train(epoches,
                    lr_all, reg_all,
                    lr_pu, lr_t, lr_beta,  
                    reg_pu, reg_t, reg_beta)
        times.append(time.clock() - start)
        
        rmse_p.append(np.sqrt(np.mean((model.P - P)**2)))
        rmse_t.append(np.sqrt(np.mean((model.t - t)**2)))
        rmse_beta.append(np.sqrt(np.mean((model.beta - beta)**2)))
        
        e_beta1.append(model.error_beta1)
        e_beta2.append(model.error_beta2)
        e_beta3.append(model.error_beta3)
        e_bigger.append(model.error_bigger)
        e_smaller.append(model.error_smaller)
        e_cdf.append(model.error_cdf)
        e_threshold.append(model.error_thresholds)
        e_x.append(model.error_x)
        e_p.append(model.error_p)
        e_prediction.append(model.error_prediction)
        e_trues.append(model.error_trues)
    # print('=======SGD=======')
    # print('time cost: \n ==> {:.4f}'.format(np.mean(times)))
    # print('RMSE of P: \n ==> {:.4f}'.format(np.mean(rmse_p)))
    # print('RMSE of t: \n ==> {:.4f}'.format(np.mean(rmse_t)))
    # print('RMSE of beta: \n ==> {:.4f}'.format(np.mean(rmse_beta)))
    # print('RMSE of overall: \n ==> {:.4f}'.format(np.mean([np.mean(rmse_p),
    #                                                        np.mean(rmse_t),
    #                                                        np.mean(rmse_beta)])))
    
    df = pd.DataFrame(index=['SGD'], columns=['w', 't', 'beta', 'overall', 'training time'])
    df.loc['SGD',:] = [np.mean(rmse_p),
                       np.mean(rmse_t),
                       np.mean(rmse_beta),
                       np.mean([np.mean(rmse_p),
                                np.mean(rmse_t),
                                np.mean(rmse_beta)]),
                       np.mean(times)]
    df.to_csv('ToConcatenate_fix_param\SGD_df%d.csv'%j)
    
    colname = [i for i in [0]]
    df_e = pd.DataFrame(columns = ['e_beta1','e_beta2','e_beta3','e_bigger','e_smaller','e_cdf',
                                 'e_threshold','e_x_input','e_p','e_prediction','e_trues'],
                        index = colname)
    
    df_e.loc[:,'e_beta1'] = e_beta1
    df_e.loc[:,'e_beta2'] = e_beta2
    df_e.loc[:,'e_beta3'] = e_beta3
    df_e.loc[:,'e_bigger'] = e_bigger
    df_e.loc[:,'e_smaller'] = e_smaller
    df_e.loc[:,'e_cdf'] = e_cdf
    df_e.loc[:,'e_threshold'] = e_threshold
    df_e.loc[:,'e_x_input'] = e_x
    df_e.loc[:,'e_p'] = e_p
    df_e.loc[:,'e_prediction'] = e_prediction
    df_e.loc[:,'e_trues'] = e_trues
    
    df_e.to_csv('ToConcatenate_fix_param\SGD_df_error%d.csv'%j)









