
# finite sample setting
# verify that SGD and ASGD converges to the minimizer
# try to extend sgd_finite.R to alpha = 0, 0.2, 0.4, 0.6, 0.8, 1.0 

# parameter
epoch = 100  #  100                  
n = 250                    
dim = 3                   
alpha = c(0, 0.2, 0.4, 0.6, 0.8,1) 

r1 = 0.02

k = n*epoch
rn = new_rn = rep(r1, length(alpha)) 

X_mean=c(5,-2,1)
X_sd=1
theta = c(3,5,2)  

len_alpha = length(alpha)

# generate data and obtain thetahat
set.seed(2020)
X = Y = c() 
for(j in 1:n){
  xj = rnorm(dim, mean = X_mean, sd = X_sd)
  yj = as.numeric(xj %*% theta + rnorm(1))
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
      sgd1[m,] = sgd1[m,] + new_rn * x %*% (y - x %*% sgd1[m,])

      sgd_store[m, kk,] = sgd1[m,]
      err_sgd[m, kk] = sqrt(mean((sgd1[m,] - theta)^2))
      
      asgd[m,] = asgd[m,] + (sgd1[m,] - asgd[m,])/kk
      err_asgd[m, kk] = sqrt(mean((asgd[m,] - theta)^2))
    } 
  }  
}


#圖4.1.1.1學習率γ_t=γ_1 t^(-α) SGD的RMSE
matplot(1:25000, t(err_sgd[1:len_alpha,]), type="l",
        main="finite sample: SGD", ylab="RMSE", xlab="iteration t",xlim=c(0,30000),xaxt='n')
axis(1,at = seq(0,25000,5000),labels = seq(0,25000,5000))
legend('topright',lty=1:6,col = 1:6,legend = paste('alpha',alpha,sep= '='))

#圖4.1.1.2學習率γ_t=γ_1 t^(-α) ASGD的RMSE
matplot(1:25000, t(err_asgd[1:len_alpha,]), type="l",
        main="finite sample: ASGD", ylab="RMSE", xlab="iteration t",xlim=c(0,30000),xaxt='n')
axis(1,at = seq(0,25000,5000),labels = seq(0,25000,5000))
legend('topright',lty=1:6,col = 1:6,legend =paste('alpha',alpha,sep= '='))


#圖4.1.1.3學習率γ_t=γ_1 t^(-α)衰退比較圖
lr_series <- r1*(matrix(rep((1:25000),6),nrow=6,byrow=T)^(-alpha))
matplot(1:500,t(lr_series[,1:500]),type='l',xlab = 'iteration t',ylab = expression(gamma[t]),
        main='learning rate with differant alpha',cex.lab=1.3)
legend('topright',lty=1:6,col = 1:6,legend = paste('alpha',alpha,sep= '='))





