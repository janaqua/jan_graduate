#ネΘ家览戈栋


simu_data <- function(S,f,m,n,N){
  P = matrix(rnorm(f*m,0,0.1),nrow = f)
  Q = matrix(rnorm(f*n,0,0.1),nrow = f)
  
  bu = matrix(rnorm(m))
  bi = matrix(rnorm(n))
  t1 = rnorm(1)
  beta  = rnorm(S-2)
  
  y = t(Q) %*% P
  y = y + matrix(rep(bu,n),nrow = n,byrow=T) + matrix(rep(bi,m),ncol = m)
  
  threshold = cumsum(c(t1,exp(beta)))
  
  cdf = array(c(rep(-Inf,m*n),rep(y,S-1),rep(Inf,m*n)),dim = c(dim(y),S+1))

  for( i in 1:(S+1)){
    if(sum(i != c(1,6))==2){
      cdf[,,i] = threshold[i-1] - cdf[,,i] 
    }
    cdf[,,i] = 1/(1+exp(-cdf[,,i]))
    
  }
  
  pdf = array(NA,dim=c(dim(y),S))
  
  for(i in 1:(S)){
    pdf[,,i] = cdf[,,i+1] - cdf[,,i]
  }
  #row is item ; col is user
  
  r = matrix(NA,nrow = n, ncol = m)
  for(i in 1:n){
    for(j in 1:m){
      r[i,j] = sample(S,1,prob = pdf[i,j,])
    }
  }
  for_data = matrix(1,nrow = n,ncol=m)
  data = data.frame('item'=as.vector(for_data*(1:n)),'user'=as.vector(t(t(for_data)*(1:m))))
  
  data['rating'] = as.vector(r)
  get <- sample(1:nrow(data),N,replace=F)
  data <- data[get,]
  param = list(P,Q,bu,bi,t1,beta)
  return(list(data,param))
  }




#SGD ASGD 

SGD_ASGD_process <- function(data,f,alpha,batch,epoches,
                             lr_p,lr_q,lr_bu,lr_bi,lr_t1,lr_beta){
  
  
  m = length(unique(data$user)) #穦计
  n = length(unique(data$item)) #坝珇计

  #﹍て把计  
  SGD_P = matrix(rnorm(f*m,0,0.1),nrow = f)
  SGD_Q = matrix(rnorm(f*n,0,0.1),nrow = f)
  
  SGD_bu =  matrix(rep(0,m))
  SGD_bi =  matrix(rep(0,n))
  SGD_t1 =  rnorm(1)
  SGD_beta =  rnorm(5-2)
  

  ASGD_P = ASGD_Q = ASGD_bu = ASGD_bi = ASGD_t1 = ASGD_beta = 0
  
  time_start = Sys.time() #璸ノ

  count = 0
  err_sgd = err_asgd = matrix(NA,nrow = epoches*nrow(data)/batch,ncol=6)
  err_sgd = as.data.frame(err_sgd)
  err_asgd = as.data.frame(err_asgd)
  colnames(err_sgd) = colnames(err_asgd) = c('P','Q','bu','bi','t1','beta')
  
  for(i in 1:epoches){
    
    time_mid = Sys.time()
      
    sample = sample(1:nrow(data),nrow(data))
    data = data[sample,]
    
    for(j in seq(1,nrow(data),batch)){
      
      #ネΘ︳璸
      x = (data[j:(j+batch-1),-3])
      
      rating = data[j:(j+batch-1),3]

      
      pred = diag(t(SGD_Q[,x[,1]]) %*% SGD_P[,x[,2]]) + SGD_bu[x[,2],] + SGD_bi[x[,1],]
      threshold = cumsum(c(SGD_t1,exp(SGD_beta)))

      
      cdf = t(1/(1+exp(t(matrix(rep(pred,5-1),ncol=(5-1)))-threshold)))
      cdf = cbind(0,cdf,1)
      
      pdf = t(apply(cdf,MARGIN = 1,diff))
      
      
      #厩策瞯癐搭
      count = count+1
      new_lr_p = min(30,lr_p*(count^-alpha))
      new_lr_q = min(30,lr_q*(count^-alpha))
      new_lr_bu = min(3,lr_bu*(count^-alpha))
      new_lr_bi = min(3,lr_bi*(count^-alpha))
      new_lr_t1 = min(1.816,lr_t1*(count^-alpha))
      new_lr_beta = min(rep(1.816,3),lr_beta*(count^-alpha))
      

      #把计穝
      bigger_r = c()
      smaller_r = c()
      for(k in 1:batch){
        bigger_r = c(bigger_r,1 - cdf[k,rating[k]+1])
        smaller_r = c(smaller_r,cdf[k,rating[k]])
      }
      
      diff_r = bigger_r - smaller_r
    

      update_P = matrix(0,nrow = nrow(SGD_P),ncol = ncol(SGD_P))
      update_Q = matrix(0,nrow = nrow(SGD_Q),ncol = ncol(SGD_Q))
      update_bu = matrix(0,nrow = nrow(SGD_bu),ncol = ncol(SGD_bu))
      update_bi = matrix(0,nrow = nrow(SGD_bi),ncol = ncol(SGD_bi))
      
      for(l in 1:nrow(x)){
        update_P[,x$user[l]] <- update_P[,x$user[l]] - SGD_Q[,x$item[l]]*diff_r[l]
        update_Q[,x$item[l]] <- update_Q[,x$item[l]] - SGD_P[,x$user[l]]*diff_r[l]
        update_bu[x$user[l],] <- update_bu[x$user[l],] - diff_r[l]
        update_bi[x$item[l],] <- update_bi[x$item[l],] - diff_r[l]
        
      }
      
      SGD_P = SGD_P + new_lr_p*update_P/batch
      SGD_Q = SGD_Q + new_lr_q*update_Q/batch
      SGD_bu = SGD_bu + new_lr_bu*update_bu/batch
      SGD_bi = SGD_bi + new_lr_bi*update_bi/batch
      
      SGD_t1 = SGD_t1 + new_lr_t1*(mean(diff_r))
      
      beta1 = SGD_beta[1] ; beta2 = SGD_beta[2] ; beta3 = SGD_beta[3]
      
      update_beta1 = -sum(cdf[rating==5,5]) + sum(diff_r[rating==4]) +
        sum(diff_r[rating ==3]) + sum(cdf[rating==2,3]*(1-cdf[rating==2,3])/(cdf[rating==2,3] - cdf[rating==2,2]))
      update_beta1 = exp(beta1) * update_beta1 / batch
      
      update_beta2 = -sum(cdf[rating==5,5]) + sum(diff_r[rating==4]) +
        sum(cdf[rating==3,4]*(1-cdf[rating==3,4])/(cdf[rating==3,4] - cdf[rating==3,3]))
      update_beta2 = exp(beta2) * update_beta2 / batch 
      
      
      update_beta3 = -sum(cdf[rating==5,5]) + 
        sum(cdf[rating==4,5]*(1-cdf[rating==4,5])/(cdf[rating==4,5] - cdf[rating==4,4]))
      update_beta3 = exp(beta3) * update_beta3 / batch
      
      update_beta = c(update_beta1,update_beta2,update_beta3)
      
      SGD_beta = SGD_beta + new_lr_beta*update_beta
      
      ASGD_beta = ASGD_beta + (SGD_beta - ASGD_beta)/count
      ASGD_t1 = ASGD_t1 + (SGD_t1 - ASGD_t1)/count
      ASGD_P = ASGD_P + (SGD_P - ASGD_P)/count
      ASGD_Q = ASGD_Q + (SGD_Q - ASGD_Q)/count
      ASGD_bu = ASGD_bu + (SGD_bu - ASGD_bu)/count
      ASGD_bi = ASGD_bi + (SGD_bi - ASGD_bi)/count
      
      err_sgd[count,] = c(sqrt(mean((SGD_P-param[[1]])^2)),sqrt(mean((SGD_Q-param[[2]])^2)),sqrt(mean((SGD_bu-param[[3]])^2))
                           ,sqrt(mean((SGD_bi-param[[4]])^2)),sqrt(mean((SGD_t1-param[[5]])^2)),sqrt(mean((SGD_beta-param[[6]])^2)))
      err_asgd[count,] = c(sqrt(mean((ASGD_P-param[[1]])^2)),sqrt(mean((ASGD_Q-param[[2]])^2)),sqrt(mean((ASGD_bu-param[[3]])^2))
                           ,sqrt(mean((ASGD_bi-param[[4]])^2)),sqrt(mean((ASGD_t1-param[[5]])^2)),sqrt(mean((ASGD_beta-param[[6]])^2)))
    }
    #璸ノ
    each_epoch_cost_time = as.numeric(difftime(Sys.time(),time_mid,units = 'secs'))
    cat('Epoches:',i,'/',epoches ,'speed:',round(each_epoch_cost_time,2), 's/it' ,'left:', ((epoches-i)*each_epoch_cost_time) %/% 60,'m',
        round(((epoches-i)*each_epoch_cost_time) %% 60,2),'s','\n')
  }
  
  SGD <- list(SGD_P,SGD_Q,SGD_bu,SGD_bi,SGD_t1,SGD_beta)
  ASGD <- list(ASGD_P,ASGD_Q,ASGD_bu,ASGD_bi,ASGD_t1,ASGD_beta)
  return(list(SGD,ASGD,err_sgd,err_asgd))
}



set.seed(2025)
S = 5 # ratings ㏕﹚5礚猭繦種э把计Α临⊿Τ快猭笆┹甶
f = 20 # latent factor
m = 100 # members
n = 500 # items 

N = 50000 # sample size
simu <- simu_data(S,f,m,n,N)
data <- simu[[1]]
param <- simu[[2]]
table(data$rating)


alpha = 0.6#

lr_p = 30
lr_p = 100^alpha*lr_p

lr_q = 30
lr_q = 100^alpha*lr_q

lr_bu = 3
lr_bu = 100^alpha*lr_bu

lr_bi = 3
lr_bi = 100^alpha*lr_bi


lr_t1 = 1.816
lr_t1 = 100^alpha*lr_t1

lr_beta = c(1.816,1.816,1.816)
lr_beta = 100^alpha*lr_beta



batch = 500
epoches = 100

result <- SGD_ASGD_process(data,f,alpha,batch,epoches,lr_p,lr_q,lr_bu ,
                           lr_bi,lr_t1,lr_beta)
param <- simu[[2]]
SGD_hat <- result[[1]]
ASGD_hat <- result[[2]]

#SGD-RMSE
matplot(1:nrow(result[[3]]),result[[3]],type = 'l',lwd=1,xlab='iteration t',ylab='RMSE',main='RMSE batch-SGD',ylim = c(0,1.8))
legend('topright',col=1:6,lty=1:6,legend=c('P','Q','bu','bi','t1','beta'))


#ASGD-RMSE
matplot(1:nrow(result[[4]]),result[[4]],type = 'l',lwd=1,xlab='iteration t',ylab='RMSE',main='RMSE batch-ASGD',ylim = c(0,1.8))
legend('topright',col=1:6,lty=1:6,legend=c('P','Q','bu','bi','t1','beta'))


##predict SGD
y = t(SGD_hat[[2]]) %*% SGD_hat[[1]]  + matrix(rep(SGD_hat[[3]],n),nrow = n,byrow=T) + matrix(rep(SGD_hat[[4]],m),ncol = m)

threshold = cumsum(c(SGD_hat[[5]],exp(SGD_hat[[6]])))

cdf = array(c(rep(-Inf,m*n),rep(y,S-1),rep(Inf,m*n)),dim = c(dim(y),S+1))
for( i in 1:(S+1)){
  if(sum(i != c(1,6))==2){
    cdf[,,i] = threshold[i-1] - cdf[,,i] 
  }
  cdf[,,i] = 1/(1+exp(-cdf[,,i]))
  
}
pdf = array(NA,dim=c(dim(y),S))

for(i in 1:(S)){
  pdf[,,i] = cdf[,,i+1] - cdf[,,i]
}

r = matrix(NA,nrow = n, ncol = m)
for(i in 1:n){
  for(j in 1:m){
    r[i,j] = which.max(pdf[i,j,])
  }
}

true = data[order(data$user,data$item),]
#RMSE
sqrt(mean((true[,3]- as.vector(r))^2))

#Precision
sum(true[,3] == as.vector(r))/length(r)

table(true[,3],as.vector(r))




#predict ASGD
y = t(ASGD_hat[[2]]) %*% ASGD_hat[[1]]  + matrix(rep(ASGD_hat[[3]],n),nrow = n,byrow=T) + matrix(rep(ASGD_hat[[4]],m),ncol = m)

threshold = cumsum(c(ASGD_hat[[5]],exp(ASGD_hat[[6]])))

cdf = array(c(rep(-Inf,m*n),rep(y,S-1),rep(Inf,m*n)),dim = c(dim(y),S+1))
for( i in 1:(S+1)){
  if(sum(i != c(1,6))==2){
    cdf[,,i] = threshold[i-1] - cdf[,,i] 
  }
  cdf[,,i] = 1/(1+exp(-cdf[,,i]))
  
}
pdf = array(NA,dim=c(dim(y),S))

for(i in 1:(S)){
  pdf[,,i] = cdf[,,i+1] - cdf[,,i]
}

r = matrix(NA,nrow = n, ncol = m)
for(i in 1:n){
  for(j in 1:m){
    r[i,j] = which.max(pdf[i,j,])
  }
}

true = data[order(data$user,data$item),]

#RMSE
sqrt(mean((true[,3]- as.vector(r))^2))

#Precision
sum(true[,3] == as.vector(r))/length(r)
table(true[,3],as.vector(r))


#true parameter
y = t(param[[2]]) %*% param[[1]]  + matrix(rep(param[[3]],n),nrow = n,byrow=T) + matrix(rep(param[[4]],m),ncol = m)

threshold = cumsum(c(param[[5]],exp(param[[6]])))

cdf = array(c(rep(-Inf,m*n),rep(y,S-1),rep(Inf,m*n)),dim = c(dim(y),S+1))
for( i in 1:(S+1)){
  if(sum(i != c(1,6))==2){
    cdf[,,i] = threshold[i-1] - cdf[,,i] 
  }
  cdf[,,i] = 1/(1+exp(-cdf[,,i]))
  
}
pdf = array(NA,dim=c(dim(y),S))

for(i in 1:(S)){
  pdf[,,i] = cdf[,,i+1] - cdf[,,i]
}

r = matrix(NA,nrow = n, ncol = m)
for(i in 1:n){
  for(j in 1:m){
    r[i,j] = which.max(pdf[i,j,])
  }
}
true = data[order(data$user,data$item),]

#RMSE
sqrt(mean((true[,3]- as.vector(r))^2))

#Precision
sum(true[,3] == as.vector(r))/length(r)
table(true[,3],as.vector(r))

