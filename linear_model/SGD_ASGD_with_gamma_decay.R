set.seed(2020)

#SGD and ASGD with gamma decay
lr = 0.001
beta <- rnorm(5)
ASGD <- c()
SGD <- c()
n = 1000
alpha = 0.6
m=100
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


ASGD_simu <- (ASGD - beta)*sqrt(n)
ASGD_simu_cov  <- cov(t(ASGD_simu))
ASGD_simu_cov

SGD_simu <- (SGD - beta)*sqrt(n)
SGD_simu_cov  <- cov(t(SGD_simu))
SGD_simu_cov


B <- rep(5,5) %*% t(rep(5,5)) + diag(1,5)
B_inv <- solve(B)
B_inv
sum(sqrt((ASGD_simu_cov - B_inv)^2/length(B_inv)))
abs(sum((ASGD_simu_cov - B_inv)/B_inv))/length(B_inv)

sum(sqrt((SGD_simu_cov - B_inv)^2/length(B_inv)))
abs(sum((SGD_simu_cov - B_inv)/B_inv))/length(B_inv)
