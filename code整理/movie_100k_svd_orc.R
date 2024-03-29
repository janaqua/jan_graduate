#movie lens 100k

data <- read.csv('D:\\User\\Desktop\\\\河\\Thesis\\code俱瞶\\ml-100k\\u.data',sep='',header=F)
colnames(data) <- c('user','item','rating','timestamp')
data <- data[-4]
#describe statistic 

library(ggplot2)
ggplot(data=data)+geom_bar(aes(x=rating,fill=as.factor(rating)))+guides(fill=F)



#train test 

set.seed(2021)
get_train <- sample(nrow(data),round(0.8*nrow(data)),replace = F)
train <- data[get_train,]
test <- data[-get_train,]

ctrl <- 0

get_u <- setdiff(test$user,train$user)
get_i <- setdiff(test$item,train$item)

while(ctrl==0){
  tmp <- NULL
  for(i in get_u){
    tmp <- rbind(tmp,test[test$user==i,])
    test <- test[!(test$user==i),]
  }
  for(i in get_i){
    tmp <- rbind(tmp,test[test$item==i,])
    test <- test[!(test$item==i),]
  }
  num <- nrow(tmp)
  train <- rbind(train,tmp)
  out <- sample(nrow(train),num,replace = F)
  
  train <- train[-out,]
  out_train <- train[out,]

  test <- rbind(test,out_train)
  get_u <- setdiff(test$user,train$user)
  get_i <- setdiff(test$item,train$item)
  if(length(get_i)==0 & length(get_u)==0){
    ctrl = 1
  }
}


#SVD++ based model 
SGD_ASGD_process <- function(data,f,alpha,batch,epoches,
                             lr_p,lr_q,lr_bu,lr_bi,lr_t1,lr_beta){
  
  
  m = length(unique(data$user)) #穦计
  n = length(unique(data$item)) #坝珇计
  
  user = unique(data$user)
  user = data.frame(index=c(1:m),row.names = user)
  
  item = unique(data$item)
  item = data.frame(index=c(1:n),row.names = item)
  

  #﹍て把计  
  SGD_P = matrix(rnorm(f*m,0,0.1),nrow = f)
  SGD_Q = matrix(rnorm(f*n,0,0.1),nrow = f)
  
  SGD_bu =  matrix(rep(0,m))
  SGD_bi =  matrix(rep(0,n))
  SGD_t1 =  rnorm(1)
  SGD_beta =  rnorm(5-2)
  
  
  ASGD_P = ASGD_Q = ASGD_bu = ASGD_bi = ASGD_t1 = ASGD_beta =  0
  
  time_start = Sys.time() #璸ノ
  
  count = 0
  
  
  for(i in 1:epoches){
    
    time_mid = Sys.time()
    
    sample = sample(1:nrow(data),nrow(data))
    data = data[sample,]
    
    for(j in seq(1,nrow(data),batch)){
      
      #ネΘ︳璸
      x = (data[j:(j+batch-1),-3])
      
      rating = data[j:(j+batch-1),3]
      
      batch_user <- user[as.character(x[,'user']),]
      batch_item <- item[as.character(x[,'item']),]

      
      pred = colSums(SGD_Q[,batch_item] * SGD_P[,batch_user] ) +
        SGD_bu[batch_user,] + SGD_bi[batch_item,]
      threshold = cumsum(c(SGD_t1,exp(SGD_beta)))
      
      
      cdf = t(1/(1+exp(t(matrix(rep(pred,5-1),ncol=(5-1)))-threshold)))
      cdf = cbind(0,cdf,1)
      
      pdf = t(apply(cdf,MARGIN = 1,diff))
      
      #厩策瞯癐搭
      count = count+1
      new_lr_p = min(lr_p,100^alpha*lr_p*(count^-alpha))
      new_lr_q = min(lr_q,100^alpha*lr_q*(count^-alpha))
      
      new_lr_bu = min(lr_bu,100^alpha*lr_bu*(count^-alpha))
      new_lr_bi = min(lr_bi,100^alpha*lr_bi*(count^-alpha))
      new_lr_t1 = min(lr_t1,100^alpha*lr_t1*(count^-alpha))
      new_lr_beta = min(rep(lr_beta,3),100^alpha*lr_beta*(count^-alpha))
      
      
      #把计穝
      bigger_r = c()
      smaller_r = c()
      for(k in 1:batch){
        bigger_r = c(bigger_r,1 - cdf[k,rating[k]+1])
        smaller_r = c(smaller_r,cdf[k,rating[k]])
      }
      
      diff_r = bigger_r - smaller_r
      
      
      update_P = matrix(0,nrow = nrow(SGD_P),ncol = ncol(SGD_P))
      update_Q = matrix(0,nrow = nrow(SGD_Q),ncol = ncol(SGD_Q))
      
      update_bu = matrix(0,nrow = nrow(SGD_bu),ncol = ncol(SGD_bu))
      update_bi = matrix(0,nrow = nrow(SGD_bi),ncol = ncol(SGD_bi))
      
      
      for(l in 1:nrow(x)){
        update_P[,batch_user[l]] <- update_P[,batch_user[l]] - 
          SGD_Q[,batch_item[l]]*diff_r[l]
        
        update_Q[,batch_item[l]] <- update_Q[,batch_item[l]] - 
          SGD_P[,batch_user[l]]*diff_r[l]
        
        update_bu[batch_user[l],] <- update_bu[batch_user[l],] - diff_r[l]
        update_bi[batch_item[l],] <- update_bi[batch_item[l],] - diff_r[l]
        
      }
      
      
      SGD_P = SGD_P + new_lr_p*update_P/batch
      SGD_Q = SGD_Q + new_lr_q*update_Q/batch
      SGD_bu = SGD_bu + new_lr_bu*update_bu/batch
      SGD_bi = SGD_bi + new_lr_bi*update_bi/batch
      
      
      SGD_t1 = SGD_t1 + new_lr_t1*(mean(diff_r))
      
      beta1 = SGD_beta[1] ; beta2 = SGD_beta[2] ; beta3 = SGD_beta[3]
      
      update_beta1 = -sum(cdf[rating==5,5]) + sum(diff_r[rating==4]) +
        sum(diff_r[rating ==3]) + sum(cdf[rating==2,3]*(1-cdf[rating==2,3])/(cdf[rating==2,3] - cdf[rating==2,2]))
      update_beta1 = exp(beta1) * update_beta1 / batch
      
      update_beta2 = -sum(cdf[rating==5,5]) + sum(diff_r[rating==4]) +
        sum(cdf[rating==3,4]*(1-cdf[rating==3,4])/(cdf[rating==3,4] - cdf[rating==3,3]))
      update_beta2 = exp(beta2) * update_beta2 / batch 
      
      
      update_beta3 = -sum(cdf[rating==5,5]) + 
        sum(cdf[rating==4,5]*(1-cdf[rating==4,5])/(cdf[rating==4,5] - cdf[rating==4,4]))
      update_beta3 = exp(beta3) * update_beta3 / batch
      
      update_beta = c(update_beta1,update_beta2,update_beta3)
      
      SGD_beta = SGD_beta + new_lr_beta*update_beta
      
      ASGD_beta = ASGD_beta + (SGD_beta - ASGD_beta)/count
      ASGD_t1 = ASGD_t1 + (SGD_t1 - ASGD_t1)/count
      ASGD_P = ASGD_P + (SGD_P - ASGD_P)/count
      ASGD_Q = ASGD_Q + (SGD_Q - ASGD_Q)/count
      ASGD_bu = ASGD_bu + (SGD_bu - ASGD_bu)/count
      ASGD_bi = ASGD_bi + (SGD_bi - ASGD_bi)/count
      
      
    }
    #璸ノ
    each_epoch_cost_time = as.numeric(difftime(Sys.time(),time_mid,units = 'secs'))
    cat('Epoches:',i,'/',epoches ,'speed:',round(each_epoch_cost_time,2), 's/it' ,'left:', ((epoches-i)*each_epoch_cost_time) %/% 60,'m',
        round(((epoches-i)*each_epoch_cost_time) %% 60,2),'s','\n')
  }
  
  SGD <- list(SGD_P,SGD_Q,SGD_bu,SGD_bi,SGD_t1,SGD_beta)
  ASGD <- list(ASGD_P,ASGD_Q,ASGD_bu,ASGD_bi,ASGD_t1,ASGD_beta)
  return(list(SGD,ASGD,user,item))
}



alpha = 0.6

lr_p = 10

lr_q = 10

lr_bu = 3


lr_bi = 3

lr_t1 = 1.816

lr_beta = c(1.816,1.816,1.816)

batch = 500
epoches = 20

f = 50 # latent factor
result <- SGD_ASGD_process(train,f,alpha,batch,epoches,lr_p,lr_q,lr_bu ,
                           lr_bi,lr_t1,lr_beta)



#f 100
f = 100 # latent factor
result2 <- SGD_ASGD_process(train,f,alpha,batch,epoches,lr_p,lr_q,lr_bu ,
                            lr_bi,lr_t1,lr_beta)

#f 200
f = 200 # latent factor
result3 <- SGD_ASGD_process(train,f,alpha,batch,epoches,lr_p,lr_q,lr_bu ,
                            lr_bi,lr_t1,lr_beta)



# predict 50

BatchSGD <- result[[1]]
user <- result[[3]]
item <- result[[4]]


y = t(BatchSGD[[2]]) %*% (BatchSGD[[1]])  +
  matrix(rep(BatchSGD[[3]],nrow(item)),nrow = nrow(item),byrow=T) + matrix(rep(BatchSGD[[4]],nrow(user)),ncol = nrow(user))

threshold = cumsum(c(BatchSGD[[5]],exp(BatchSGD[[6]])))

cdf = array(c(rep(-Inf,nrow(user)*nrow(item)),rep(y,5-1),rep(Inf,nrow(user)*nrow(item))),dim = c(dim(y),5+1))
for( i in 1:(5+1)){
  if(sum(i != c(1,6))==2){
    cdf[,,i] = threshold[i-1] - cdf[,,i] 
  }
  cdf[,,i] = 1/(1+exp(-cdf[,,i]))
  
}
pdf = array(NA,dim=c(dim(y),5))

for(i in 1:(5)){
  pdf[,,i] = cdf[,,i+1] - cdf[,,i]
}

r = matrix(NA,nrow = nrow(item), ncol = nrow(user))
for(i in 1:nrow(item)){
  for(j in 1:nrow(user)){
    #r[i,j] = which.max(pdf[i,j,])
    r[i,j] = which.max(pdf[i,j,])
  }
}

train['pred'] <- NA
for(i in 1:nrow(train)){
  train[i,'pred'] <- r[item[as.character(train[i,'item']),],user[as.character(train[i,'user']),]]
}

#RMSE 
RMSE_50_train <- sqrt(mean((train$rating-train$pred)^2))

#FCP
nc <- 0 
nd <- 0 


for(i in unique(train$user)){
  tmp_r <- train[train$user==i,'rating']
  tmp_pred <- train[train$user==i,'pred']
  for(j in 1:length(tmp_r)){
    for(k in 1:length(tmp_r)){
      cond1 = (tmp_r[j] > tmp_r[k]) & (tmp_pred[j] > tmp_pred[k])
      #      cond2 = (tmp_r[j] == tmp_r[k]) & (tmp_pred[j] == tmp_pred[k])
      #      cond3 = (tmp_r[j] < tmp_r[k]) & (tmp_pred[j] < tmp_pred[k])
      #      if(cond1 | cond2 | cond3){nc = nc+1}
      if(cond1){nc = nc+1}
      
      cond1 = (tmp_r[j] >= tmp_r[k]) & (tmp_pred[j] < tmp_pred[k])
      #      cond2 = (tmp_r[j] <= tmp_r[k]) & (tmp_pred[j] > tmp_pred[k])
      #      if(cond1|cond2){nd = nd +1}
      if(cond1){nd = nd +1}
    }
  }
}
FCP_50_train <- nc/(nc+nd)


test['pred'] <- NA
for(i in 1:nrow(test)){
  test[i,'pred'] <- r[item[as.character(test[i,'item']),],user[as.character(test[i,'user']),]]
}
#RMSE 
RMSE_50_test <- sqrt(mean((test$rating-test$pred)^2))


for(i in unique(test$user)){
  tmp_r <- test[test$user==i,'rating']
  tmp_pred <- test[test$user==i,'pred']
  for(j in 1:length(tmp_r)){
    for(k in 1:length(tmp_r)){
      cond1 = (tmp_r[j] > tmp_r[k]) & (tmp_pred[j] > tmp_pred[k])
      #      cond2 = (tmp_r[j] == tmp_r[k]) & (tmp_pred[j] == tmp_pred[k])
      #      cond3 = (tmp_r[j] < tmp_r[k]) & (tmp_pred[j] < tmp_pred[k])
      #      if(cond1 | cond2 | cond3){nc = nc+1}
      if(cond1){nc = nc+1}
      
      cond1 = (tmp_r[j] >= tmp_r[k]) & (tmp_pred[j] < tmp_pred[k])
      #      cond2 = (tmp_r[j] <= tmp_r[k]) & (tmp_pred[j] > tmp_pred[k])
      #      if(cond1|cond2){nd = nd +1}
      if(cond1){nd = nd +1}
      
    }
  }
}
FCP_50_test <- nc/(nc+nd)

# predict 100

BatchSGD <- result2[[1]]
user <- result2[[3]]
item <- result2[[4]]


y = t(BatchSGD[[2]]) %*% (BatchSGD[[1]])  +
  matrix(rep(BatchSGD[[3]],nrow(item)),nrow = nrow(item),byrow=T) + matrix(rep(BatchSGD[[4]],nrow(user)),ncol = nrow(user))

threshold = cumsum(c(BatchSGD[[5]],exp(BatchSGD[[6]])))

cdf = array(c(rep(-Inf,nrow(user)*nrow(item)),rep(y,5-1),rep(Inf,nrow(user)*nrow(item))),dim = c(dim(y),5+1))
for( i in 1:(5+1)){
  if(sum(i != c(1,6))==2){
    cdf[,,i] = threshold[i-1] - cdf[,,i] 
  }
  cdf[,,i] = 1/(1+exp(-cdf[,,i]))
  
}
pdf = array(NA,dim=c(dim(y),5))

for(i in 1:(5)){
  pdf[,,i] = cdf[,,i+1] - cdf[,,i]
}

r = matrix(NA,nrow = nrow(item), ncol = nrow(user))
for(i in 1:nrow(item)){
  for(j in 1:nrow(user)){
    #r[i,j] = which.max(pdf[i,j,])
    r[i,j] = which.max(pdf[i,j,])
  }
}

train['pred'] <- NA
for(i in 1:nrow(train)){
  train[i,'pred'] <- r[item[as.character(train[i,'item']),],user[as.character(train[i,'user']),]]
}

#RMSE 
RMSE_100_train <- sqrt(mean((train$rating-train$pred)^2))

#FCP
nc <- 0 
nd <- 0 


for(i in unique(train$user)){
  tmp_r <- train[train$user==i,'rating']
  tmp_pred <- train[train$user==i,'pred']
  for(j in 1:length(tmp_r)){
    for(k in 1:length(tmp_r)){
      cond1 = (tmp_r[j] > tmp_r[k]) & (tmp_pred[j] > tmp_pred[k])
      #      cond2 = (tmp_r[j] == tmp_r[k]) & (tmp_pred[j] == tmp_pred[k])
      #      cond3 = (tmp_r[j] < tmp_r[k]) & (tmp_pred[j] < tmp_pred[k])
      #      if(cond1 | cond2 | cond3){nc = nc+1}
      if(cond1){nc = nc+1}
      
      cond1 = (tmp_r[j] >= tmp_r[k]) & (tmp_pred[j] < tmp_pred[k])
      #      cond2 = (tmp_r[j] <= tmp_r[k]) & (tmp_pred[j] > tmp_pred[k])
      #      if(cond1|cond2){nd = nd +1}
      if(cond1){nd = nd +1}
    }
  }
}
FCP_100_train <- nc/(nc+nd)


test['pred'] <- NA
for(i in 1:nrow(test)){
  test[i,'pred'] <- r[item[as.character(test[i,'item']),],user[as.character(test[i,'user']),]]
}
#RMSE 
RMSE_100_test <- sqrt(mean((test$rating-test$pred)^2))



for(i in unique(test$user)){
  tmp_r <- test[test$user==i,'rating']
  tmp_pred <- test[test$user==i,'pred']
  for(j in 1:length(tmp_r)){
    for(k in 1:length(tmp_r)){
      cond1 = (tmp_r[j] > tmp_r[k]) & (tmp_pred[j] > tmp_pred[k])
      #      cond2 = (tmp_r[j] == tmp_r[k]) & (tmp_pred[j] == tmp_pred[k])
      #      cond3 = (tmp_r[j] < tmp_r[k]) & (tmp_pred[j] < tmp_pred[k])
      #      if(cond1 | cond2 | cond3){nc = nc+1}
      if(cond1){nc = nc+1}
      
      cond1 = (tmp_r[j] >= tmp_r[k]) & (tmp_pred[j] < tmp_pred[k])
      #      cond2 = (tmp_r[j] <= tmp_r[k]) & (tmp_pred[j] > tmp_pred[k])
      #      if(cond1|cond2){nd = nd +1}
      if(cond1){nd = nd +1}
      
    }
  }
}
FCP_100_test <- nc/(nc+nd)

# predict 200

BatchSGD <- result3[[1]]
user <- result3[[3]]
item <- result3[[4]]


y = t(BatchSGD[[2]]) %*% (BatchSGD[[1]])  +
  matrix(rep(BatchSGD[[3]],nrow(item)),nrow = nrow(item),byrow=T) + matrix(rep(BatchSGD[[4]],nrow(user)),ncol = nrow(user))

threshold = cumsum(c(BatchSGD[[5]],exp(BatchSGD[[6]])))

cdf = array(c(rep(-Inf,nrow(user)*nrow(item)),rep(y,5-1),rep(Inf,nrow(user)*nrow(item))),dim = c(dim(y),5+1))
for( i in 1:(5+1)){
  if(sum(i != c(1,6))==2){
    cdf[,,i] = threshold[i-1] - cdf[,,i] 
  }
  cdf[,,i] = 1/(1+exp(-cdf[,,i]))
  
}
pdf = array(NA,dim=c(dim(y),5))

for(i in 1:(5)){
  pdf[,,i] = cdf[,,i+1] - cdf[,,i]
}

r = matrix(NA,nrow = nrow(item), ncol = nrow(user))
for(i in 1:nrow(item)){
  for(j in 1:nrow(user)){
    #r[i,j] = which.max(pdf[i,j,])
    r[i,j] = which.max(pdf[i,j,])
  }
}

train['pred'] <- NA
for(i in 1:nrow(train)){
  train[i,'pred'] <- r[item[as.character(train[i,'item']),],user[as.character(train[i,'user']),]]
}

#RMSE 
RMSE_200_train <- sqrt(mean((train$rating-train$pred)^2))

#FCP
nc <- 0 
nd <- 0 


for(i in unique(train$user)){
  tmp_r <- train[train$user==i,'rating']
  tmp_pred <- train[train$user==i,'pred']
  for(j in 1:length(tmp_r)){
    for(k in 1:length(tmp_r)){
      cond1 = (tmp_r[j] > tmp_r[k]) & (tmp_pred[j] > tmp_pred[k])
      #      cond2 = (tmp_r[j] == tmp_r[k]) & (tmp_pred[j] == tmp_pred[k])
      #      cond3 = (tmp_r[j] < tmp_r[k]) & (tmp_pred[j] < tmp_pred[k])
      #      if(cond1 | cond2 | cond3){nc = nc+1}
      if(cond1){nc = nc+1}
      
      cond1 = (tmp_r[j] >= tmp_r[k]) & (tmp_pred[j] < tmp_pred[k])
      #      cond2 = (tmp_r[j] <= tmp_r[k]) & (tmp_pred[j] > tmp_pred[k])
      #      if(cond1|cond2){nd = nd +1}
      if(cond1){nd = nd +1}
    }
  }
}
FCP_2000_train <- nc/(nc+nd)


test['pred'] <- NA
for(i in 1:nrow(test)){
  test[i,'pred'] <- r[item[as.character(test[i,'item']),],user[as.character(test[i,'user']),]]
}
#RMSE 
RMSE_200_test <- sqrt(mean((test$rating-test$pred)^2))


for(i in unique(test$user)){
  tmp_r <- test[test$user==i,'rating']
  tmp_pred <- test[test$user==i,'pred']
  for(j in 1:length(tmp_r)){
    for(k in 1:length(tmp_r)){
      cond1 = (tmp_r[j] > tmp_r[k]) & (tmp_pred[j] > tmp_pred[k])
      #      cond2 = (tmp_r[j] == tmp_r[k]) & (tmp_pred[j] == tmp_pred[k])
      #      cond3 = (tmp_r[j] < tmp_r[k]) & (tmp_pred[j] < tmp_pred[k])
      #      if(cond1 | cond2 | cond3){nc = nc+1}
      if(cond1){nc = nc+1}
      
      cond1 = (tmp_r[j] >= tmp_r[k]) & (tmp_pred[j] < tmp_pred[k])
      #      cond2 = (tmp_r[j] <= tmp_r[k]) & (tmp_pred[j] > tmp_pred[k])
      #      if(cond1|cond2){nd = nd +1}
      if(cond1){nd = nd +1}
      
    }
  }
}
FCP_200_test <- nc/(nc+nd)

