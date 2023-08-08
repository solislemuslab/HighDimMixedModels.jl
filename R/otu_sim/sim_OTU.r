sim_OTU=function(N,p,q,ni,beta,sigma,theta,dis,par_dis,phi){
  ntot <- sum(ni) # total number of observations
  grp <- factor(rep(1:N,times=ni)) # grouping variable
  x <- matrix(0, ntot, p)  # initialize a matrix of zeros
  for (i in 1:p) {
    x[, i] <- sample.h(ntot,phi, distri = dis,r=par_dis[1],alpha1=par_dis[2],alpha2=par_dis[3])
  }
  ##########################################################
  if (q==1){
    bi1 <- rep(rnorm(N,0,theta[1]),times=ni)  
    bi <- rbind(bi1)
  }else if (q==2){
    bi1 <- rep(rnorm(N,0,theta[1]),times=ni) 
    bi2 <- rep(rnorm(N,0,theta[2]),times=ni)
    bi <- rbind(bi1,bi2)
  }else if (q==3){
    bi1 <- rep(rnorm(N,0,theta[1]),times=ni) 
    bi2 <- rep(rnorm(N,0,theta[2]),times=ni)
    bi3 <- rep(rnorm(N,0,theta[3]),times=ni)
    bi <- rbind(bi1,bi2,bi3)
  }else{
    bi1 <- rep(rnorm(N,0,theta[1]),times=ni) 
    bi2 <- rep(rnorm(N,0,theta[2]),times=ni)
    bi3 <- rep(rnorm(N,0,theta[3]),times=ni)
    bi.rest <- matrix(0,nrow = q-3,ncol = ntot)
    bi <- rbind(bi1,bi2,bi3,bi.rest)
  }
  
  x <- cbind(1,x) 
  z <- x[,1:q,drop=FALSE]
  y <- numeric(ntot)
  for (k in 1:ntot) 
    y[k] <- x[k,]%*%beta + t(z[k,])%*%bi[,grp[k]] +  rnorm(1, 0, sigma)
  return(list(y=y,x=x,z=z,grp=grp))
  }