# verify Simulation for explicit SGD in Section 4.1.1 of Toulis and Airoldi; under stream setting
# with gamma1 = 1.2, 2.5, 10 etc
# if r_n = r_1/n, the result is worse. The paper suggest modified r_n

m = 150 # number of replicates                    
n = 1500    # sample size                    
dim = 20                  
g1 = 5 # 2.5 # 1.2         
k = n/100
ind = (1:k)*100

# set.seed(2020)

alpha = 1
X_mean = rep(0,dim)
X_sd = sqrt(runif(dim,0.5,5))
beta = rep(1,dim)  
gamma1 = rep(g1,dim) 
         
sgd_store = array(0,dim=c(m, dim, k))
    
for (i in 1:m){
    sgd = rnorm(dim, mean = 0,sd = 1)
    SGD = c()
      
    for (j in 1:n){
        
        xj = rnorm(dim,mean = X_mean,sd = X_sd)
        yj = as.numeric(xj %*% beta + rnorm(1))

        gammaj = min(0.3, gamma1/(j^(alpha)+sum(xj^2)))
        # gammaj = gamma1/j
        # gammaj = gamma1/(j^(alpha)+sum(xj*xj))
        sgd = sgd + gammaj* xj %*% (yj - xj %*% sgd)
        SGD = cbind(SGD, sgd)
    }
    
    sgd_store[i,,] = SGD[,ind]
}
sgd   # sensitive to the choice of (g1, alpha)

S = X_mean %*% t(X_mean) + diag(X_sd^2)
V = g1^2*solve(2*g1*S-diag(1,dim)) %*% S
sumDiagV = sum(diag(V))

trV = c()
err = c()
for (i in 1:k){
    Vk = (100*i)*cov(sgd_store[,,i])
    trV = rbind(trV, c(sumDiagV, sum(diag(Vk))))
    err = c(err, norm(V-Vk, type = "F"))
    print(c(sumDiagV, sum(diag(Vk))))
}

matplot(1:k, trV, type="l", xlab="t (sample size=100*t)") 
matplot(1:k, log(trV), type="l", xlab="t (sample size=100*t)")
err
