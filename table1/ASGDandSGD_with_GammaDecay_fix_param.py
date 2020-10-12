from OptimizationMethods import SGD_GammaDecay
import time
import pickle
import numpy as np
import pandas as pd
from tqdm import  tqdm

for j in tqdm([1,2,5,10,20,30,50]) : 
    lr_all = 1.6
    reg_all = 0.001
    lr_pu, lr_t, lr_beta = 2.1, 2.8, np.array([1.11, 1.052, 1.05])  
    reg_pu, reg_t, reg_beta = 0., 0., 0.
    epoches = j
    
    times = []
    rmse_p = []
    rmse_t = []
    rmse_beta = []
    
    rmse_avg_p = []
    rmse_avg_t = []
    rmse_avg_beta = []

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
        
        model = SGD_GammaDecay.ord_reg(data)
        start = time.clock()
        model.train(epoches, 
                    lr_all, reg_all,
                    lr_pu, lr_t, lr_beta,  
                    reg_pu, reg_t, reg_beta)
        times.append(time.clock() - start)
        
        P_avg = np.mean(model.pu_list, axis=0)
        t_avg = np.mean(model.t_list, axis=0)
        beta_avg = np.mean(model.beta_list, axis=0)
        
        rmse_p.append(np.sqrt(np.mean((model.P - P)**2)))
        rmse_t.append(np.sqrt(np.mean((model.t - t)**2)))
        rmse_beta.append(np.sqrt(np.mean((model.beta - beta)**2)))
         
        rmse_avg_p.append(np.sqrt(np.mean((P_avg - P)**2)))
        rmse_avg_t.append(np.sqrt(np.mean((t_avg - t)**2)))
        rmse_avg_beta.append(np.sqrt(np.mean((beta_avg - beta)**2)))
        
            
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
    # print('=======SGD with gamma_n decayed=======')
    # print('time cost: \n ==> {:.4f} seconds'.format(np.mean(times)))
    # print('RMSE of P: \n ==> {:.4f}'.format(np.mean(rmse_p)))
    # print('RMSE of t: \n ==> {:.4f}'.format(np.mean(rmse_t)))
    # print('RMSE of beta: \n ==> {:.4f}'.format(np.mean(rmse_beta)))
    # print('RMSE of overall: \n ==> {:.4f}'.format(np.mean([np.mean(rmse_p),
    #                                                        np.mean(rmse_t),
    #                                                        np.mean(rmse_beta)])))
    
    # print('=======ASGD with gamma_n decayed=======')
    # print('time cost: \n ==> {:.4f} seconds'.format(np.mean(times)))
    # print('RMSE of P: \n ==> {:.4f}'.format(np.mean(rmse_avg_p)))
    # print('RMSE of t: \n ==> {:.4f}'.format(np.mean(rmse_avg_t)))
    # print('RMSE of beta: \n ==> {:.4f}'.format(np.mean(rmse_avg_beta)))
    # print('RMSE of overall: \n ==> {:.4f}'.format(np.mean([np.mean(rmse_avg_p),
    #                                                        np.mean(rmse_avg_t),
    #                                                        np.mean(rmse_avg_beta)])))
    
    
    
    df = pd.DataFrame(index=['SGD with gamma_n decayed', 'ASGD with gamma_n decayed'], columns=['w', 't', 'beta', 'overall', 'training time'])
    df.loc['SGD with gamma_n decayed',:] = [np.mean(rmse_p),
                                            np.mean(rmse_t),
                                            np.mean(rmse_beta),
                                            np.mean([np.mean(rmse_p),
                                                     np.mean(rmse_t),
                                                     np.mean(rmse_beta)]),
                                            np.mean(times)]
    
    df.loc['ASGD with gamma_n decayed',:] = [np.mean(rmse_avg_p),
                                            np.mean(rmse_avg_t),
                                            np.mean(rmse_avg_beta),
                                            np.mean([np.mean(rmse_avg_p),
                                                     np.mean(rmse_avg_t),
                                                     np.mean(rmse_avg_beta)]),
                                            np.mean(times)]
    
    
    df.to_csv('ToConcatenate_fix_param\ASGDandSGD_with_GammaDecay_df%d.csv'%j)

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
    df_e.to_csv('ToConcatenate_fix_param\ASGDandSGD_with_GammaDecay_df_error%d.csv'%j)
