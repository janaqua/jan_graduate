#---------------------------------計時器--------------------------------#



time_count <- function(time_start,time_epoch_start,now,total,each=1){
  
  if(now%%each ==0){
    
    time_epoch_cost = as.numeric(difftime(Sys.time() , time_epoch_start , units = 'secs'))
    total_time_cost = round(as.numeric(difftime(Sys.time(),time_start,units = 'secs')))
    left_time = round((total-now) * (time_epoch_cost))
    
    cat('Epoch:',now,'/',total,'   ')
    cat('Time: ', total_time_cost %/% 3600,'h:', total_time_cost %/% 60 %% 60,'m:',total_time_cost %% 60 ,
        's < ', left_time %/% 3600,'h:',left_time %/% 60 %% 60,'m:',left_time %% 60,'s , ',
        round(time_epoch_cost^-1,2),' it/s \n',sep='')
    
  }
  
}



#-------------------------ASGD process function------------------------#



ASGD <- function(m,dim,n,lr_beta,  # m :估計值個數 / dim :變數維度 / n :資料筆數 / lr_beta:學習率
                 alpha=0.6,is_gamma_decay=F,    # alpha /  is_gamma_decay :是否要進行gamma_decaym，預設否
                 X_mean=5,X_sd=1,B_mean=0,B_sd=1, # X,B~Normal,X平均，X標準差，Beta平均，Beta標準差
                 is_count_time=F  # is_count_time : 是否要計算運算速度，預設否
                 ){   

  
    beta = rnorm(dim,B_mean,B_sd)  #模擬實際beta值
    lr_beta = new_lr_beta = rep(lr_beta,dim) #設定起始學習率，預設每個beta一樣
    
    ASGD_beta_hat = c() #建立ASGD估計值存放空間
    
    
    time_start = Sys.time()   #計時用
  
  for(i in 1:m){
    
    #設定beta起始估計值
      beta_hat = rnorm(dim,mean = 0,sd = 1)
      beta_hat_store = c()
      
      time_epoch_start = Sys.time() #計時用
    
    #ASGD迭代
    
      for(j in 1:n){
        
        # gamma_decay
          if(is_gamma_decay==T){
            new_lr_beta = lr_beta*(j^(-alpha))
            }
        
        #模擬X分配且產生Y
          X = rnorm(dim,mean = X_mean,sd = X_sd)
          Y = as.numeric(X %*% beta + rnorm(1))

        #迭代beta估計值
          gradient = X * as.numeric(Y - X %*% beta_hat)
          beta_hat = beta_hat + new_lr_beta*gradient
        
        
        beta_hat_store = cbind(beta_hat_store,beta_hat)
      }
    
    #儲存本次參數估計值
      ASGD_beta_hat = cbind(ASGD_beta_hat,rowMeans(beta_hat_store))
    

    # 計時器
      if(is_count_time==T){
        time_count(time_start,time_epoch_start,i,m,each=1)
      }
    }
    
    return(list(beta,ASGD_beta_hat))
}




#----------------------------設定起始值並執行程序---------------------#


m = 1000                    #估計值個數
n = 200                     #資料大小(n) = 迭代總次數(t) 
dim = 5                     #決定維度
lr_beta = 0.01              #學習率

set.seed(2020)



# 預設X~N(5,1) B~N(0,1) 不進行gamma_decay 不計算運行速度



result <- ASGD(m,dim,n,lr_beta)

True_beta <- result[[1]]         #真實beta
ASGD_beta_hat <- result[[2]]     #估計的beta



#估計值轉換Normal分配並秀出共變異數矩陣


Trans_beta_to_normal <- (ASGD_beta_hat - True_beta)*sqrt(n)
cov(t(Trans_beta_to_normal))


#計算理論共變異數矩陣


B <- rep(5,dim) %*% t(rep(5,dim)) + diag(1,5)
B_inv <- solve(B)
B_inv
