# finite sample setting
# verify that SGD and ASGD converges to the minimizer
# try to extend sgd_finite.R to alpha = 0, 0.2, 0.4, 0.6, 0.8, 1.0 

# parameter
epoch = 100  #  100                  
n = 250                    
dim = 3                   
r1 = 0.01           

k = n*epoch
rn = new_rn = rep(r1, dim) 

alpha = c(0, 0.2, 0.4, 0.6, 0.8, 1.0) 
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
betahat = solve(t(X) %*% X) %*% t(X) %*% Y

sgd = rnorm(dim, mean = 0, sd = 1); Gam_j = diag(1,dim);
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
        
      new_rn = rn*(kk^(-alpha[m]))
      sgd1[m,] = sgd1[m,] + new_rn * x %*% (y - x %*% sgd1[m,])
      # print(c(kk,sgd1[m,]))
      sgd_store[m, kk,] = sgd1[m,]
      err_sgd[m, kk] = mean((sgd1[m,] - betahat)^2)

      # asgd = colMeans(sgd_store[m,,],na.rm = T)
      asgd[m,] = asgd[m,] + (sgd1[m,] - asgd[m,])/kk
      err_asgd[m, kk] = mean((asgd[m,] - betahat)^2)
    } 
  }  
}


str(sgd_store)
matplot(1:k, cbind(err_sgd[1,], err_sgd[4,], err_asgd[1,], err_asgd[4,]), type="l", main="finite sample: sgd(black)/sgd_0.6(r)/asgd(g)/asgd_0.6(blue)", ylab="err", xlab="k")

matplot(1:k, t(err_sgd[1:len_alpha,]), type="l", main="finite sample: sgd", ylab="err", xlab="k")
matplot(1:k, t(err_asgd[1:len_alpha,]), type="l", main="finite sample: asgd", ylab="err", xlab="k")

matplot(1:5000, t(err_sgd[1:len_alpha,20001:25000]), type="l", main="finite sample: sgd", ylab="err", xlab="k")
matplot(1:5000, cbind(err_sgd[1,20001:25000], err_asgd[1,20001:25000]), type="l", main="finite sample: sgd", ylab="err", xlab="k")
