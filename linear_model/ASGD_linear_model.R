#---------------------------------�p�ɾ�--------------------------------#



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



ASGD <- function(m,dim,n,lr_beta,  # m :���p�ȭӼ� / dim :�ܼƺ��� / n :��Ƶ��� / lr_beta:�ǲ߲v
                 alpha=0.6,is_gamma_decay=F,    # alpha /  is_gamma_decay :�O�_�n�i��gamma_decaym�A�w�]�_
                 X_mean=5,X_sd=1,B_mean=0,B_sd=1, # X,B~Normal,X�����AX�зǮt�ABeta�����ABeta�зǮt
                 is_count_time=F  # is_count_time : �O�_�n�p��B��t�סA�w�]�_
                 ){   

  
    beta = rnorm(dim,B_mean,B_sd)  #�������beta��
    lr_beta = new_lr_beta = rep(lr_beta,dim) #�]�w�_�l�ǲ߲v�A�w�]�C��beta�@��
    
    ASGD_beta_hat = c() #�إ�ASGD���p�Ȧs��Ŷ�
    
    
    time_start = Sys.time()   #�p�ɥ�
  
  for(i in 1:m){
    
    #�]�wbeta�_�l���p��
      beta_hat = rnorm(dim,mean = 0,sd = 1)
      beta_hat_store = c()
      
      time_epoch_start = Sys.time() #�p�ɥ�
    
    #ASGD���N
    
      for(j in 1:n){
        
        # gamma_decay
          if(is_gamma_decay==T){
            new_lr_beta = lr_beta*(j^(-alpha))
            }
        
        #����X���t�B����Y
          X = rnorm(dim,mean = X_mean,sd = X_sd)
          Y = as.numeric(X %*% beta + rnorm(1))

        #���Nbeta���p��
          gradient = X * as.numeric(Y - X %*% beta_hat)
          beta_hat = beta_hat + new_lr_beta*gradient
        
        
        beta_hat_store = cbind(beta_hat_store,beta_hat)
      }
    
    #�x�s�����ѼƦ��p��
      ASGD_beta_hat = cbind(ASGD_beta_hat,rowMeans(beta_hat_store))
    

    # �p�ɾ�
      if(is_count_time==T){
        time_count(time_start,time_epoch_start,i,m,each=1)
      }
    }
    
    return(list(beta,ASGD_beta_hat))
}




#----------------------------�]�w�_�l�Ȩð���{��---------------------#


m = 1000                    #���p�ȭӼ�
n = 200                     #��Ƥj�p(n) = ���N�`����(t) 
dim = 5                     #�M�w����
lr_beta = 0.01              #�ǲ߲v

set.seed(2020)



# �w�]X~N(5,1) B~N(0,1) ���i��gamma_decay ���p��B��t��



result <- ASGD(m,dim,n,lr_beta)

True_beta <- result[[1]]         #�u��beta
ASGD_beta_hat <- result[[2]]     #���p��beta



#���p���ഫNormal���t�èq�X�@�ܲ��Ưx�}


Trans_beta_to_normal <- (ASGD_beta_hat - True_beta)*sqrt(n)
cov(t(Trans_beta_to_normal))


#�p��z�צ@�ܲ��Ưx�}


B <- rep(5,dim) %*% t(rep(5,dim)) + diag(1,5)
B_inv <- solve(B)
B_inv