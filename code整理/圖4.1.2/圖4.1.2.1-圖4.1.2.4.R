
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

sgd = rnorm(dim, mean = 0, sd = 1)

sgd_store = array(NA, dim=c(len_alpha, n, dim))
err_sgd = err_asgd = array(0, dim=c(len_alpha, n))
asgd = array(0, dim=c(len_alpha, dim))

sgd1 = rep(1,len_alpha) %*% t(sgd)
for(j in 1:n){
  
  x =  rnorm(dim, mean = X_mean, sd = X_sd)
  y =  as.numeric(x %*% beta + rnorm(1))
  
  for (m in 1:len_alpha){
    
    new_rn = rn[m]*(j^(-alpha[m]))
    new_rn = min(c(0.02,new_rn))
    sgd1[m,] = sgd1[m,] + new_rn * x %*% (y - x %*% sgd1[m,])
    
    sgd_store[m, j,] = sgd1[m,]
    err_sgd[m, j] = sqrt(mean((sgd1[m,] - beta)^2))
    
    asgd[m,] = asgd[m,] + (sgd1[m,] - asgd[m,])/j
    err_asgd[m, j] = sqrt(mean((asgd[m,] - beta)^2))
  } 
}  


#瓜4.1.2.1玡1000ΩSGDのASGD把计︳璸RMSE
matplot(1:1000,cbind(err_sgd[1,1:1000],err_asgd[1,1:1000]),type='l',main='stream sample: SGD & ASGD',ylab='RMSE',
        xlab = 'iteration t',lwd=1.5,cex.lab =1.2, cex.axis = 1.2)
legend('topright',lwd=1.5,lty=1:6,col = 1:2,legend = c('SGD','ASGD'),cex = 1.2)



#瓜4.1.2.224000Ω穝SGD把计︳璸RMSE
matplot(1001:25000, t(err_sgd[1:len_alpha,1001:25000]), type="l",
        main="stream sample: SGD", ylab="RMSE", xlab="iteration t",xaxt='n',xlim = c(1000,30000))
axis(1,at = c(1000,seq(5000,25000,5000)),labels = c(1000,seq(5000,25000,5000)))
legend('topright',lty=1:6,col = 1:6,legend = paste('alpha',alpha,sep= '='))

#瓜4.1.2.324000Ω穝ASGD把计︳璸RMSE
matplot(1:24000, t(err_asgd[1:len_alpha,1001:25000]), type="l",
        main="stream sample: ASGD", ylab="RMSE", xlab="iteration t",xlim=c(0,30000),xaxt='n')
axis(1,at = c(1000,seq(5000,25000,5000)),labels = c(1000,seq(5000,25000,5000)))
legend('topright',lty=1:6,col = 1:6,legend = paste('alpha',alpha,sep= '='))


#瓜4.1.2.4 =15000ΩSGDのASGDぇRMSEゑ耕瓜
matplot(1:5000,cbind(err_sgd[len_alpha,20001:25000],err_asgd[len_alpha,20001:25000]),type='l',
        main = expression('stream sample: '~alpha~'='~1~',SGD(black),ASGD(red)'),
        ylab = 'RMSE',xlab='iteration t',xaxt='n')
axis(1,at=seq(0,5000,1000),labels=seq(20000,25000,1000))
legend('topright',lty=1:2,col=1:6,legend = c('SGD','ASGD'))




