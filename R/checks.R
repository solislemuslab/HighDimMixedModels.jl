library("splmm")
library("penalized")

#Simulated data as found here: https://rdrr.io/cran/lmmlasso/man/lmmlasso.html
set.seed(54)
N <- 20           # number of groups
p <- 6            # number of covariates (not including intercept)
q <- 2            # number of random effect covariates
ni <- rep(6,N)    # observations per group
ntot <- sum(ni)   # total number of observations

grp <- factor(rep(1:N,times=ni)) # grouping variable

beta <- c(1,2,4,3,0,0,0) # fixed-effects coefficients
x <- cbind(1,matrix(rnorm(ntot*p),nrow=ntot)) # design matrix

bi1 <- rep(rnorm(N,0,3),times=ni) 
bi2 <- rep(rnorm(N,0,2),times=ni)
bi <- rbind(bi1,bi2) # Psi=diag(3^2,2^2)

z <- x[,1:2,drop=FALSE]

y <- numeric(ntot)
for (k in 1:ntot) y[k] <- x[k,]%*%beta + t(z[k,])%*%bi[,grp[k]] + rnorm(1)
#sigma is 1

ygrp <- split(y, grp)
xgrp <- split.data.frame(x, grp)
zgrp <- split.data.frame(z, grp)
zidgrp <- mapply(splmm:::ZIdentity, zgrp)

###Checking initial values
init <- optL1(y, x[, -1], model = "linear", fold = 10, trace = FALSE)
betaStart <- c(init$fullfit@unpenalized, init$fullfit@penalized)
cov_start <- splmm:::covStartingValues(xgrp, ygrp, zgrp, zidgrp, betaStart, ntot, N)

####Fits 

# wrong random effects structure
fit2 <- splmm(x=x,y=y,z=z,grp=grp,lambda=10,pdMat="pdIdent")
summary(fit2)
plot(fit2)

# correct random effects structure
fit3 <- lmmlasso(x=x,y=y,z=z,grp=grp,lambda=10,pdMat="pdDiag")
summary(fit3)
plot(fit3)