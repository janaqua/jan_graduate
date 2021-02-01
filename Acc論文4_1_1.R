# verify Simulation for explicit SGD in Section 4.1.1 of Toulis and Airoldi; under stream setting
# with gamma1 = 1.2, 2.5, 10 etc
# if r_n = r_1/n, the result is worse. The paper suggest modified r_n

m = 150 # number of replicates                    
n = 1500    # sample size                    
dim = 20                  
g1 = seq(1.2,10,length.out = 25)# 2.5 # 1.2         



# set.seed(2020)
trV = c()

for(g in g1){

alpha = 1
X_mean = rep(0,dim)
X_sd = sqrt(runif(dim,0.5,5))

beta = rep(1,dim)  
gamma1 = rep(g,dim) 

sgd_store = array(0,dim=c(m, dim, 1))
isgd_store = array(0,dim=c(m, dim, 1))
asgd_store = c()
for (i in 1:m){
  sgd = rnorm(dim, mean = 0,sd = 1)
  isgd = rnorm(dim, mean = 0,sd = 1)
  

  SGD = c()
  
  for (j in 1:n){
    
    xj = rnorm(dim,mean = X_mean,sd = X_sd)
    yj = as.numeric(xj %*% beta + rnorm(1))
    
    gammaj = min(0.025, gamma1/(j^(alpha)+sum(xj^2)))
    # gammaj = gamma1/j
    # gammaj = gamma1/(j^(alpha)+sum(xj*xj))
    sgd = sgd + gammaj* xj %*% (yj - xj %*% sgd)
    #sgd =  (diag(dim)-gammaj*((xj)%*% t(xj)))  %*% sgd  + gammaj*yj*xj
    isgd  = solve(diag(dim)+gammaj*((xj)%*% t(xj))) %*% isgd + (gammaj *solve(diag(dim)+gammaj*((xj)%*% t(xj))) *yj) %*% xj
    SGD = cbind(SGD, sgd)
  }
  
  sgd_store[i,,] = sgd
  isgd_store[i,,] = isgd
  asgd_store = rbind(asgd_store,rowMeans(SGD))
}
#sgd   # sensitive to the choice of (g1, alpha)

S = X_mean %*% t(X_mean) + diag(X_sd^2)
V = (g^2)*solve(2*g*S-diag(dim)) %*% S
sumDiagV = sum(diag(V))


Vk = cov(sgd_store[,,1])*n
Vki = cov(isgd_store[,,1])*n
Vka = cov(asgd_store)*n
trV = rbind(trV, c(sumDiagV, sum(diag(Vk)),sum(diag(Vki)),sum(diag(Vka))))

}
matplot(1:k, trV, type="l", xlab="t (sample size=100*t)") 
matplot(g1, log(trV), type="l", xlab="learning rate",ylim=c(2.5,7))
legend('topleft',col=1:4,legend = c('SGD_theoretical','SGD','SGD_implicit','ASGD'),lty = c(1,2,2,2))
err

eigen(S)
log(trV)

log(sumDiagV)
