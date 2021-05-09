# verify Simulation for explicit SGD in Section 4.1.1 of Toulis and Airoldi; under stream setting
# with gamma1 = 1.2, 2.5, 10 etc
# if r_n = r_1/n, the result is worse. The paper suggest modified r_n

m = 150 # number of replicates                    
n = 25000    # sample size                    
dim = 20                  
g1 = seq(1.2,10,length.out = 25)# 2.5 # 1.2         



set.seed(2021)
trV = c()

for(g in g1){

  alpha = 1
  X_mean = rep(0,dim)
  X_sd = sqrt(runif(dim,0.5,5))
  
  beta = rep(1,dim)  
  gamma1 = rep(g,dim) 
  
  sgd_store = array(0,dim=c(m, dim, 1))
  asgd_store = array(0,dim=c(m, dim, 1))
  
  for (i in 1:m){
    sgd = rnorm(dim, mean = 0,sd = 1)
  
    SGD = c()
    asgd = 0
    for (j in 1:n){
      
      xj = rnorm(dim,mean = X_mean,sd = X_sd)
      yj = as.numeric(xj %*% beta + rnorm(1))
      
      gammaj = min(0.02, gamma1/(j^(alpha)+sum(xj^2)))
      
      sgd = sgd + gammaj* xj %*% (yj - xj %*% sgd)
      asgd = asgd + (sgd - asgd)/j
    }
    
    sgd_store[i,,] = sgd
    asgd_store[i,,] = asgd
    
  }
  #sgd   # sensitive to the choice of (g1, alpha)
  
  S = X_mean %*% t(X_mean) + diag(X_sd^2)
  V = (g^2)*solve(2*g*S-diag(dim)) %*% S
  sumDiagV = sum(diag(V))
  
  
  Vk = cov(sgd_store[,,1])*n
  Vka = cov(asgd_store[,,1])*n
  trV = rbind(trV, c(sumDiagV, sum(diag(Vk)),sum(diag(Vka))))
  cat('lr = ',g,'\n')
}

#圖4.1.2.9 SGD及ASGD方法在不同學習率下的變化
ntrV <- cbind(sum(diag(solve(S))),trV) #add ASGD_theoretical
matplot(g1, log(ntrV), type="l", xlab="learning rate",lwd=2,ylab = 'log(trace)',main='stream data',ylim=c(2,5))
legend('topleft',col=1:4,legend = c('ASGD_theoretical','SGD_theoretical','SGD','ASGD'),lty = 1:4,lwd=2)
