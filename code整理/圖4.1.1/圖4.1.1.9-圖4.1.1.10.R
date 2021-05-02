
# finite sample setting
# verify that SGD and ASGD converges to the minimizer
# try to extend sgd_finite.R to alpha = 0, 0.2, 0.4, 0.6, 0.8, 1.0 

# parameter
epoch = 100  #  100                  
retry = 150
n = 250                    
dim = 3                   
alpha = c(0, 0.2, 0.4, 0.6, 0.8,1) 

r1 = 0.02


k = n*epoch
rn = new_rn = rep(r1, length(alpha)) 

X_mean=c(5,-2,1)
X_sd=1
beta = c(3,5,2)  
len_alpha = length(alpha)

# generate data and obtain betahat
set.seed(2020)
X = Y = c() 
for(j in 1:n){
  xj = rnorm(dim, mean = X_mean, sd = X_sd)
  yj = as.numeric(xj %*% beta + rnorm(1))
  X = rbind(X, xj)
  Y = c(Y,yj)
}


sgd_store = array(NA, dim=c(len_alpha, retry, k, dim))
asgd = array(0, dim=c(len_alpha, retry, k, dim))


for(l in 1:retry){
  sgd = rnorm(dim, mean = 0, sd = 1)
  sgd1 = rep(1,len_alpha) %*% t(sgd)
  
  for (i in 1:epoch){
    
    
    ind = sample(1:n, replace=F)
    for(j in 1:n){
      
      x = X[ind[j],]
      y = Y[ind[j]]
      
      kk = (i-1)*n + j
      
      for (m in 1:len_alpha){
        
        new_rn = rn[m]*(kk^(-alpha[m]))
        sgd1[m,] = sgd1[m,] + new_rn * x %*% (y - x %*% sgd1[m,])
        sgd_store[m, l, kk,] = sgd1[m,]
        
        if(kk!=1){
          asgd[m, l, kk,] = asgd[m, l, kk-1,] + (sgd1[m,] - asgd[m, l, kk-1,])/kk
        }
        else{
          
          asgd[m, l, kk,] = (sgd1[m,])/kk
        }
      } 
    }  
    
  }
  cat('m=',l,'\n')
}

# 計算RMSE
compare <- solve(X_mean %*% t(X_mean) +diag(dim))

sgd_err = asgd_err <- array(NA,dim = c(25000,len_alpha))
for(i in 1:len_alpha){
  for(j in 1:25000){
    sgd_err[j,i] <- sqrt(mean(((cov( (t(t(sgd_store[i,,j,]) - beta))*sqrt(j)) )- compare)**2))
    asgd_err[j,i] <-sqrt(mean(((cov( (t(t(asgd[i,,j,]) - beta))*sqrt(j)) ) - compare)**2))
  }
  
}

#圖4.1.1.9 學習率γ_t=γ_1 t^(-α)且α???(0,0.5)之ASGD的估計變異RMSE
matplot(1:25000,asgd_err[,1:3],type='l',main=expression('finite sample: ASGD'~alpha~'=0, 0.2,0.4'),
        ylab='RMSE',xlab='iteration t')
legend('topright',lwd=1.5,lty=1:6,col = 1:6,legend = paste('alpha',alpha[1:3],sep= '='),cex = 1.2)

#圖4.1.1.10 學習率γ_t=γ_1 t^(-α)且α???(0.5,1)之ASGD的估計變異RMSE
matplot(1:25000,asgd_err[,4:6],col=4:6,type='l',main=expression('finite sample: ASGD'~alpha~'=0.6, 0.8,1'),
        ylab='RMSE',xlab='iteration t')
legend('topleft',lwd=1.5,lty=1:6,col = 4:6,legend = paste('alpha',alpha[4:6],sep= '='),cex = 1.2)
 



