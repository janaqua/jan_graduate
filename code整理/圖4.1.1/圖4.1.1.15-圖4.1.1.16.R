# verify Simulation for explicit SGD in Section 4.1.1 of Toulis and Airoldi; under stream setting
# with gamma1 = 1.2, 2.5, 10 etc
# if r_n = r_1/n, the result is worse. The paper suggest modified r_n

m = 150 # number of replicates         
epoch = 100
n = 250    # sample size                    
dim = 20                  
g1 = seq(1.2,10,length.out = 25)# 2.5 # 1.2         



set.seed(2021)
trV = c()

for(g in g1){
  
  X_mean = rep(0,dim)
  X_sd = sqrt(runif(dim,0.5,5))
  
  beta = rep(1,dim)  
  gamma1 = rep(g,dim) 
  
  sgd_store = array(0,dim=c(m, dim, 1))
  asgd_store = array(0,dim=c(m, dim, 1))
  
  X = Y = c() 
  for(j in 1:n){
    xj = rnorm(dim, mean = X_mean, sd = X_sd)
    yj = as.numeric(xj %*% beta + rnorm(1))
    X = rbind(X, xj)
    Y = c(Y,yj)
  }
  for (i in 1:m){
    
    sgd = sgd2 = rnorm(dim, mean = 0,sd = 1)
    
    asgd = 0
    
    for(k in 1:epoch){
      ind = sample(1:n, replace=F)
      for (j in 1:n){
        x = X[ind[j],]
        y = Y[ind[j]]
        kk = (k-1)*n + j
        
        gammaj = gamma1*kk^(-1)
        sgd = sgd + gammaj* x %*% (y - x %*% sgd)
        
        
        gammaj = gamma1*kk^(-0.6)
        sgd2 = sgd2 + gammaj* x %*% (y - x %*% sgd2)
        asgd = asgd + (sgd2 - asgd)/kk
      }
    }
    sgd_store[i,,] = sgd
    asgd_store[i,,] = asgd
  }
  
  #sgd   # sensitive to the choice of (g1, alpha)
  
  S = X_mean %*% t(X_mean) + diag(X_sd^2)
  V = (g^2)*solve(2*g*S-diag(dim)) %*% S
  sumDiagV = sum(diag(V))
  
  
  Vk = cov(sgd_store[,,1])*25000
  Vka = cov(asgd_store[,,1])*25000
  trV = rbind(trV, c(sumDiagV, sum(diag(Vk)),sum(diag(Vka))))
  cat('lr = ',g,'\n')
}


#圖4.1.1.15 SGD及ASGD方法在不同學習率下的變化

ntrV <- cbind(sum(diag(solve(S))),trV) #add ASGD_theoretical

matplot(g1, log(ntrV), type="l", xlab=expression('parameter'~gamma[1]),lwd=2,ylab = 'log(trV)',main='finite data')

legend('topright',col=1:4,legend = c('ASGD_theoretical','SGD_theoretical','SGD','ASGD'),lty = 1:4,lwd=2)



#圖4.1.1.16
# verify Simulation for explicit SGD in Section 4.1.1 of Toulis and Airoldi; under stream setting
# with gamma1 = 1.2, 2.5, 10 etc
# if r_n = r_1/n, the result is worse. The paper suggest modified r_n

m = 150 # number of replicates         
epoch = 100
n = 250    # sample size                    
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
  
  X = Y = c() 
  for(j in 1:n){
    xj = rnorm(dim, mean = X_mean, sd = X_sd)
    yj = as.numeric(xj %*% beta + rnorm(1))
    X = rbind(X, xj)
    Y = c(Y,yj)
  }
  for (i in 1:m){
    
    sgd = rnorm(dim, mean = 0,sd = 1)
    asgd = 0
    
    for(k in 1:epoch){
      ind = sample(1:n, replace=F)
      for (j in 1:n){
        x = X[ind[j],]
        y = Y[ind[j]]
        kk = (k-1)*n + j
  
        gammaj = min(0.02, gamma1/(kk^(alpha)+sum(x^2)))
        sgd = sgd + gammaj* x %*% (y - x %*% sgd)
        asgd = asgd + (sgd - asgd)/kk
      }
    }
    sgd_store[i,,] = sgd
    asgd_store[i,,] = asgd
  }
  
  #sgd   # sensitive to the choice of (g1, alpha)
  
  S = X_mean %*% t(X_mean) + diag(X_sd^2)
  V = (g^2)*solve(2*g*S-diag(dim)) %*% S
  sumDiagV = sum(diag(V))
  
  
  Vk = cov(sgd_store[,,1])*25000
  Vka = cov(asgd_store[,,1])*25000
  trV = rbind(trV, c(sumDiagV, sum(diag(Vk)),sum(diag(Vka))))
  cat('lr = ',g,'\n')
}


#圖4.1.1.16 SGD及ASGD方法在不同學習率下的變化

ntrV <- cbind(sum(diag(solve(S))),trV) #add ASGD_theoretical

matplot(g1, log(ntrV), type="l", xlab=expression('parameter'~gamma[1]),lwd=2,ylab = 'log(trV)',main='finite data')

legend('bottomright',col=1:4,legend = c('ASGD_theoretical','SGD_theoretical','SGD','ASGD'),lty = 1:4,lwd=2)

