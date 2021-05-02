#generate_data

N = 50000 #the num of observation
dim = 2 #w
r = 5 #rating

set.seed(0130)

X = matrix(rnorm(N*dim),ncol=dim)
w = rep(1,dim)
t = rnorm(1) #first threshold
beta = rnorm(r-2)

threshold = cumsum(c(t,exp(beta)))
y = X %*% w

cdf = t(1/(1+exp(t(matrix(rep(y,r-1),ncol=(r-1)))-threshold)))

cdf = cbind(0,cdf,1)

pdf = t(apply(cdf,MARGIN = 1,diff))
rating = c()
for(i in 1:nrow(pdf)){

    rating = c(rating,sample(1:ncol(pdf),1,prob = pdf[i,]))

}
table(rating)

data <- data.frame(X,rating)
param <- list(w,t,beta)


#SGD,ASGD with gamma_decay  or batch



SGD_ASGD_gamma_decay <- function(data,lr_w,lr_t,lr_beta,
                                 lamb_w,lamb_t,lamb_beta,alpha,epoches,batch)
  {
  N = nrow(data)
  dim = ncol(data)-1
  r = 5 #rating
  
  first_lr_w = lr_w
  first_lr_t = lr_t
  first_lr_beta = lr_beta
  
  w_hat = matrix(rnorm(dim,sd=0.1),nrow=1)
  t_hat = rnorm(1)
  beta_hat = rnorm(3)
  
  w_hat_asgd = 0
  t_hat_asgd = 0
  beta_hat_asgd = 0
  
  n = 1 #iteration
  time_start = Sys.time() #計時用
  RMSE_SGD <- c()
  RMSE_ASGD <- c()
  train_time <- c()
  for(i in 1:epoches){
    sample = sample(1:N,N)
    data = data[sample,]
    
    time_epoch = Sys.time()
    
      for(j in seq(1,N,batch)){
      x = as.matrix(data[j:(j+batch-1),-(dim+1)])
      rating = data[j:(j+batch-1),(dim+1)]

      pred = x %*% t(w_hat)
      threshold = cumsum(c(t_hat,exp(beta_hat)))
      
      cdf = t(1/(1+exp(t(matrix(rep(pred,r-1),ncol=(r-1)))-threshold)))
      cdf = cbind(0,cdf,1)
      
      pdf = t(apply(cdf,MARGIN = 1,diff))
      
      #gamma_Decay
      lr_w = first_lr_w*(n^-alpha)
      lr_t = first_lr_t*(n^-alpha)
      lr_beta = first_lr_beta*(n^-alpha)
      n = n+1

      #iteration
      bigger_r = c()
      smaller_r = c()
      for(k in 1:batch){
        bigger_r = c(bigger_r,1 - cdf[k,rating[k]+1]) # P(ri > r)
        smaller_r = c(smaller_r,cdf[k,rating[k]]) # P(ri < r)
      }

      diff_r = bigger_r - smaller_r # P(ri > r) - p(r <= r-1)

      update_w = (t(as.matrix(diff_r)) %*% -x)/batch - lamb_w*w_hat
      update_t = (mean(diff_r)) - lamb_t*t_hat
      
      
      
      beta1 = beta_hat[1] ; beta2 = beta_hat[2] ; beta3 = beta_hat[3]

      update_beta1 = -sum(cdf[rating==5,5]) + sum(diff_r[rating==4]) +
        sum(diff_r[rating ==3]) + sum(cdf[rating==2,3]*(1-cdf[rating==2,3])/(cdf[rating==2,3] - cdf[rating==2,2]))
      update_beta1 = exp(beta1) * update_beta1 / batch - lamb_beta[1]*beta1
      
      update_beta2 = -sum(cdf[rating==5,5]) + sum(diff_r[rating==4]) +
        sum(cdf[rating==3,4]*(1-cdf[rating==3,4])/(cdf[rating==3,4] - cdf[rating==3,3]))
      update_beta2 = exp(beta2) * update_beta2 / batch - lamb_beta[2]*beta2
      

      update_beta3 = -sum(cdf[rating==5,5]) + 
        sum(cdf[rating==4,5]*(1-cdf[rating==4,5])/(cdf[rating==4,5] - cdf[rating==4,4]))
      update_beta3 = exp(beta3) * update_beta3 / batch - lamb_beta[3]*beta3
      
      update_beta = c(update_beta1,update_beta2,update_beta3)

      w_hat = w_hat + lr_w*update_w
      t_hat = t_hat + lr_t*update_t
      beta_hat = beta_hat + lr_beta*update_beta
      
      
      w_hat_asgd = w_hat_asgd + (w_hat - w_hat_asgd)/n
      t_hat_asgd = t_hat_asgd + (t_hat - t_hat_asgd)/n
      beta_hat_asgd = beta_hat_asgd + (beta_hat - beta_hat_asgd)/n
      
      }
    each_epoch_cost_time = as.numeric(difftime(Sys.time(),time_epoch,units = 'secs'))
    cat('Epoches:',i,'/',epoches ,'speed:',1/each_epoch_cost_time, 'it/s' ,'left:', ((epoches-i)*each_epoch_cost_time) %/% 60,'m',
    ((epoches-i)*each_epoch_cost_time) %% 60,'s','\n')
    
    SGD_RMSE_w <- sqrt(mean((w_hat - param[[1]])^2))
    SGD_RMSE_t <- sqrt(mean((t_hat - param[[2]])^2))
    SGD_RMSE_beta <- sqrt(mean((beta_hat - param[[3]])^2))
    
    ASGD_RMSE_w <- sqrt(mean((w_hat_asgd - param[[1]])^2))
    ASGD_RMSE_t <- sqrt(mean((t_hat_asgd - param[[2]])^2))
    ASGD_RMSE_beta <- sqrt(mean((beta_hat_asgd - param[[3]])^2))
    
    RMSE_SGD <- c(RMSE_SGD,mean(c(SGD_RMSE_w,SGD_RMSE_t,SGD_RMSE_beta)))
    RMSE_ASGD <- c(RMSE_ASGD,mean(c(ASGD_RMSE_w,ASGD_RMSE_t,ASGD_RMSE_beta)))
    train_time <- c(train_time,as.numeric(difftime(Sys.time(),time_start,units = 'secs')))
  }
  return(list(w_hat,t_hat,beta_hat,w_hat_asgd,t_hat_asgd,beta_hat_asgd,
              as.numeric(difftime(Sys.time(),time_start,units = 'secs')),
              RMSE_SGD,RMSE_ASGD,train_time))
}

#SGD
lr_w = 0.18 ; lr_t = 0.2 ; lr_beta = c(0.18,0.02,0.023) #learnging rate gamma
lamb_w = 0. ; lamb_t = 0. ; lamb_beta = c(0.,0.,0.)  # regularize coefficient
alpha = 0. #alpha = 0 means no decay
epoches = 50
batch = 1 #batch = 1 means ASGD & SGD and others mean batch

result <-  SGD_ASGD_gamma_decay(data = data,
                                lr_w = lr_w ,lr_t = lr_t,lr_beta = lr_beta,
                                lamb_w = lamb_w,lamb_t = lamb_t,lamb_beta = lamb_beta,
                                alpha = alpha,
                                epoches = epoches,
                                batch = batch
)
SGD_final <- result[[8]]
no_batch_time <- result[[10]]


#SGD ASGD with gamma decay
lr_w = 2.1 ; lr_t = 2.8 ; lr_beta = c(1.11,1.052,1.05) #learnging rate gamma
lamb_w = 0. ; lamb_t = 0. ; lamb_beta = c(0.,0.,0.)  # regularize coefficient
alpha = 0.6 #alpha = 0 means no decay
epoches = 50
batch = 1 #batch = 1 means ASGD & SGD and others mean batch

result <-  SGD_ASGD_gamma_decay(data = data,
                                lr_w = lr_w ,lr_t = lr_t,lr_beta = lr_beta,
                                lamb_w = lamb_w,lamb_t = lamb_t,lamb_beta = lamb_beta,
                                alpha = alpha,
                                epoches = epoches,
                                batch = batch
)

SGD_decay_final <- result[[8]]
ASGD_decay_final <- result[[9]]


#Batch - SGD
lr_w = 2.1 ; lr_t = 2.8 ; lr_beta = c(1.11,1.052,1.05) #learnging rate gamma
lamb_w = 0. ; lamb_t = 0. ; lamb_beta = c(0.,0.,0.)  # regularize coefficient
alpha = 0. #alpha = 0 means no decay
epoches = 50
batch = 500 #batch = 1 means ASGD & SGD and others mean batch



result <-  SGD_ASGD_gamma_decay(data = data,
                               lr_w = lr_w ,lr_t = lr_t,lr_beta = lr_beta,
                               lamb_w = lamb_w,lamb_t = lamb_t,lamb_beta = lamb_beta,
                               alpha = alpha,
                               epoches = epoches,
                               batch = batch
                               )

 

Batch_SGD_final <- result[[8]] 
Batch_time <- result[[10]]

#batch-SGD、ASGD with gamma decay
lr_w = 2.1 ; lr_t = 2.8 ; lr_beta = c(1.11,1.052,1.05) #learnging rate gamma
lamb_w = 0. ; lamb_t = 0. ; lamb_beta = c(0.,0.,0.)  # regularize coefficient
alpha = 0.6 #alpha = 0 means no decay
epoches = 50
batch = 500 #batch = 1 means ASGD & SGD and others mean batch



result <-  SGD_ASGD_gamma_decay(data = data,
                                lr_w = lr_w ,lr_t = lr_t,lr_beta = lr_beta,
                                lamb_w = lamb_w,lamb_t = lamb_t,lamb_beta = lamb_beta,
                                alpha = alpha,
                                epoches = epoches,
                                batch = batch
)



Batch_SGD_decay_final <- result[[8]]
Batch_ASGD_decay_final <- result[[9]]
Batch_SGD_decay_time <- result[[10]]



#圖4.1.3.1各式梯度下降法在不同epoch下之RMSE
matplot(1:50,cbind(SGD_final,SGD_decay_final,ASGD_decay_final,
                   Batch_SGD_final,Batch_SGD_decay_final,Batch_ASGD_decay_final),
        ylab = 'RMSE',xlab='epoch',type='l',ylim=c(0,0.8))
legend('topright',legend = c('SGD','SGD_decay','ASGD_decay','BatchSGD',
                             'Batch_SGD_decay','Batch_ASGD_decay'),col=1:6,lty = 1:6)

#圖4.1.3.2各式梯度下降法累積訓練時間(second)
matplot(1:50,cbind(no_batch_time,Batch_time),
        ylab = 'Training time(sec)',xlab='epoch',type='l')
legend('topleft',legend = c('no batch time','batch time'),col=1:2,lty = 1:2)                       
