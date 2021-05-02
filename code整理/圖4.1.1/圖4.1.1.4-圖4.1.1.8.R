
# finite sample setting
# verify that SGD and ASGD converges to the minimizer
# try to extend sgd_finite.R to alpha = 0, 0.2, 0.4, 0.6, 0.8, 1.0 

# parameter
epoch = 100  #  100                  
n = 250                    
dim = 3                   
alpha = c(0, 0.2, 0.4, 0.6, 0.8,1) 

r1 = 0.02/(1000^-alpha)

k = n*epoch
rn = new_rn = r1

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

sgd = rnorm(dim, mean = 0, sd = 1)

sgd_store = array(NA, dim=c(len_alpha, k, dim))
err_sgd = err_asgd = array(0, dim=c(len_alpha, k))
asgd = array(0, dim=c(len_alpha, dim))

sgd1 = rep(1,len_alpha) %*% t(sgd)
for (i in 1:epoch){
  
  ind = sample(1:n, replace=F)
  for(j in 1:n){
    
    x = X[ind[j],]
    y = Y[ind[j]]
    
    kk = (i-1)*n + j
    
    for (m in 1:len_alpha){
      
      new_rn = rn[m]*(kk^(-alpha[m]))
      new_rn = min(c(0.02,new_rn))
      sgd1[m,] = sgd1[m,] + new_rn * x %*% (y - x %*% sgd1[m,])
      sgd_store[m, kk,] = sgd1[m,]
      err_sgd[m, kk] = sqrt(mean((sgd1[m,] - beta)^2))
      
      asgd[m,] = asgd[m,] + (sgd1[m,] - asgd[m,])/kk
      err_asgd[m, kk] = sqrt(mean((asgd[m,] - beta)^2))
    } 
  }  
}


#圖4.1.1.4學習率調整後γ_t= (min(0.02,γ_1 t^(α))之衰減比較圖
lr_series <- r1*(matrix(rep((1:25000),6),nrow=6,byrow=T)^(-alpha))
lr_series[lr_series >= 0.02] <- 0.02
matplot(1:25000,t(lr_series),type='l',xlab = 'iteration t',ylab = expression(gamma[t]),
        main='learning rate with differant alpha',xlim = c(0,35000),xaxt='n',lwd=1.5)
legend('topright',lty=1:6,col = 1:6,legend = paste('alpha',alpha,sep= '='))
axis(1,at = seq(0,25000,5000))



#圖4.1.1.5前1000次SGD及ASGD參數估計值的RMSE
matplot(1:1000,cbind(err_sgd[1,1:1000],err_asgd[1,1:1000]),type='l',main='finite sample: SGD & ASGD',ylab='RMSE',
        xlab = 'iteration t',lwd=1.5,cex.lab =1.2, cex.axis = 1.2)
legend('topright',lwd=1.5,lty=1:6,col = 1:2,legend = c('SGD','ASGD'),cex = 1.2)



#圖4.1.1.6 後24000次更新SGD參數估計值的RMSE
matplot(1:24000, t(err_sgd[1:len_alpha,1001:25000]), type="l",
        main="finite sample: SGD", ylab="RMSE", xlab="iteration t",xlim=c(0,30000),xaxt='n')
axis(1,at = c(seq(0,24000,5000),24000),labels = c(1000,seq(5000,25000,5000)))
legend('topright',lty=1:6,col = 1:6,legend = paste('alpha',alpha,sep= '='))

#圖4.1.1.7 後24000次更新ASGD參數估計值的RMSE
matplot(1:24000, t(err_asgd[1:len_alpha,1001:25000]), type="l",
        main="finite sample: ASGD", ylab="RMSE", xlab="iteration t",xlim=c(0,30000),xaxt='n')
axis(1,at = c(seq(0,24000,5000),24000),labels = c(1000,seq(5000,25000,5000)))
legend('topright',lty=1:6,col = 1:6,legend = paste('alpha',alpha,sep= '='))

#圖4.1.1.8 α=1下，後5000次迭代SGD及ASGD之RMSE比較圖
matplot(1:5000,cbind(err_sgd[len_alpha,20001:25000],err_asgd[len_alpha,20001:25000]),type='l',
        main = expression('finite sample: '~alpha~'='~1~',SGD(black),ASGD(red)'),
        ylab = 'RMSE',xlab='iteration t',xaxt='n')
axis(1,at=seq(0,5000,1000),labels=seq(20000,25000,1000))
legend('topright',lty=1:2,col=1:6,legend = c('SGD','ASGD'))





