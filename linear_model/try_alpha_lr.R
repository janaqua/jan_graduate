
#SGD and ASGD with gamma decay
lr = 0.02
beta <- rnorm(5)
n = 50000

m=100
ASGD_RMSE <- c()
ASGD_MAPE <- c()
SGD_RMSE <- c()
SGD_MAPE <- c()

for(k in 0:10){
  alpha = 0.1*k
  ASGD <- c()
  SGD <- c()
  
  
for(i in 1:m){
  
  beta_hat = beta_hat_sum <- rnorm(5)
  new_lr = lr
  j=1
  for(j in 1:n){
    new_lr <- lr*(j^(-alpha))
    
    X <- rnorm(5,5,1)
    Y <- X %*% beta + rnorm(1)
    
    gradient <- X * c(Y - X %*% beta_hat)
    beta_hat <- beta_hat + new_lr*gradient
    
    beta_hat_sum <- beta_hat+beta_hat_sum
    j= j+1
  }
  ASGD <- cbind(ASGD,beta_hat_sum/(n+1))
  SGD <- cbind(SGD,beta_hat)
  if(i %% 10 ==0){
    print(i)
  }
}
  ASGD_MAPE <- c(ASGD_MAPE,mean(abs((ASGD - beta)/beta)))
  ASGD_RMSE <- c(ASGD_RMSE,sqrt(mean((ASGD - beta)^2)))

  SGD_MAPE <- c(SGD_MAPE,mean(abs((SGD - beta)/beta)))
  SGD_RMSE <- c(SGD_RMSE,sqrt(mean((SGD - beta)^2)))
  }

round(ASGD_RMSE,4)
round(ASGD_MAPE,4)



