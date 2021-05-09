retry = 150
n = 25000                    
dim = 3                   
alpha = c(0, 0.2, 0.4, 0.6, 0.8,1) 

r1 = 0.02/(1000^-alpha)

rn = new_rn = r1
X_mean=c(5,-2,1)
X_sd=1
beta = c(3,5,2)  
len_alpha = length(alpha)

# generate data and obtain betahat
set.seed(2020)


sgd_store = array(NA, dim=c(len_alpha, retry, n, dim))
asgd = array(0, dim=c(len_alpha, retry, n, dim))


for(l in 1:retry){
  sgd = rnorm(dim, mean = 0, sd = 1)
  sgd1 = rep(1,len_alpha) %*% t(sgd)
  
  for(j in 1:n){
    
    x =  rnorm(dim, mean = X_mean, sd = X_sd)
    y =  as.numeric(x %*% beta + rnorm(1))
    
    kk = j
    
    for (m in 1:len_alpha){
      
      new_rn = rn[m]*(kk^(-alpha[m]))
      new_rn = min(c(0.02,new_rn))
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
  cat('m = ',l,'\n')
}

#RMSE計算
compare <- solve(X_mean %*% t(X_mean) +diag(dim))

sgd_err = asgd_err <- array(NA,dim = c(25000,len_alpha))
for(i in 1:len_alpha){
  for(j in 1:25000){
    sgd_err[j,i] <- sqrt(mean(((cov( (t(t(sgd_store[i,,j,]) - beta))*sqrt(j)) )- compare)**2))
    asgd_err[j,i] <-sqrt(mean(((cov( (t(t(asgd[i,,j,]) - beta))*sqrt(j)) ) - compare)**2))
  }
  
}

#圖4.1.2.5 stream data前1000次更新ASGD的估計變異RMSE
matplot(1:1000,asgd_err[1:1000,],type='l',main='stream sample: ASGD',ylab='RMSE',col=1,xlab='iteration t')

#圖4.1.2.6 stream data 後24000次更新ASGD的估計變異RMSE
matplot(1001:25000,asgd_err[1001:25000,],type='l',main='stream sample: ASGD',ylab='RMSE',xlab='iteration t',xaxt='n')
legend('topright',lwd=1.5,lty=1:6,col = 1:6,legend = paste('alpha',alpha,sep= '='),cex = 1.2)
axis(1,at = c(1000,seq(0,25000,5000)),labels = c(1000,seq(0,25000,5000)))


#圖4.1.2.7 學習率調整後前1000次更新ASGD的估計變異矩陣trace


sgd_trV = asgd_trV <- array(NA,dim = c(25000,len_alpha))
for(i in 1:len_alpha){
  for(j in 1:25000){
    sgd_trV[j,i] <- sum(diag(cov( (t(t(sgd_store[i,,j,]) - beta))*sqrt(j)) ))
    asgd_trV[j,i] <- sum(diag(cov( (t(t(asgd[i,,j,]) - beta))*sqrt(j))))
  }
  
}

matplot(1:1000,asgd_trV[1:1000,1],type='l',main='stream sample: ASGD trace',ylab='trace',xlab='iteration t')
abline(h=sum(diag(compare)),lwd=2,lty=1,col=1)
text(x = 200,y=sum(diag(compare))+2,labels = 'trace(V)')

#圖4.1.2.8 學習率調整後後24000次更新ASGD的估計變異矩陣trace

matplot(1:24000,asgd_trV[1001:25000,],type='l',main='stream sample: ASGD trace',ylab='trace',xlab='iteration t',xaxt='n',ylim=c(2,12.5))
axis(1,at = c(seq(0,24000,5000),24000),labels = c(1000,seq(5000,25000,5000)))
legend('topright',lwd=1.5,lty=1:6,col = 1:6,legend = paste('alpha',alpha,sep= '='),cex = 1.2)
abline(h=sum(diag(compare)),lwd=2,lty=1,col=1)
text(x = 15000,y=sum(diag(compare))+0.3,labels = 'trace(V)')

