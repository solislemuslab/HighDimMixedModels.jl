library(glue)
library(penalized)
library(emulator)
source("R/lmmSCAD_og/helpers.R")
lmmSCAD <- function(x,y,z=x,grp,weights=NULL,lambda,pdMat,nonpen=1:dim(z)[[2]], SCADa=3.7)
{
  # x, y, z = Data matrices as per teh notation in our manuscript
  # grp = group indicator (factor)
  # weights = weight vector
  # lambda = regularization parameter $\lambda$
  # pdMat = "pdIdent" for Psi = theta^2 Identity
  # pdMat = "pdDiag" for Psi = Diag(theta^2)
  # SCADa = tuning paramter a in teh SCAD penaly (default = 3.7)
  # nonpen = indices of the variables which are not to be penalized

  
  ##################### ----- controls ... --------
  tol=10^(-2); trace=1; maxIter=100; maxArmijo=20; number=5; a_init=1; delta=0.1; rho=0.001; gamma=0; 
  lower=10^(-6); upper=10^8;
  #seed=532;
  VarInt=c(0,1); CovInt=c(-5,5); thres=10^(-4);
  ranInd=1:dim(z)[[2]];

  
  ##################### calculate weights (NA: not penalized, 0: drop) -----------
  if (missing(weights))
  {
    weights <- rep(1,dim(x)[[2]])
  } else
  {
    nonpen <- which(is.na(weights))
    remove <- weights==0
    remove[nonpen] <- FALSE
    
    x <- x[,!remove,drop=FALSE]
    
    rem <- which(weights==0)
    z <- z[,!(ranInd%in%rem),drop=FALSE]
    
    weights <- weights[!remove]
    nonpen <- which(is.na(weights))
    
    ran1 <- logical(dim(x)[[2]])
    ran1[ranInd] <- TRUE
    ran2 <- ran1[!remove]
    ranInd <- which(ran2)
  }
  
  
  ##################### crucial allocations --------------------
  grp <- factor(grp)
  N <- length(levels(grp)) # N is the number of groups
  p <- dim(x)[[2]]         # p is the number of covariates
  q <- dim(z)[[2]]         # q is the number of random effects variables
  ntot <- length(y)        # ntot is the total number of observations
  ll1 <- 1/2*ntot*log(2*pi) # constant in teh objective function based on N
  
  # save the grouped information as components of a list
  yGrp <- split(y,grp)
  xGrp <- split.data.frame(x,grp)
  zGrp <- split.data.frame(z,grp)
  zIdGrp <- mapply(ZIdentity,zGrp)
  
  
  ##################### ---  Calculate objective function for the starting values -----
  init <- optL1(y,x[,-1],model="linear",fold=10,trace=FALSE)
  betaStart <- c(init$fullfit@unpenalized,init$fullfit@penalized)
  covStart <- covStartingValues(xGrp,yGrp,zGrp,zIdGrp,betaStart,ntot,N)
  sStart <- covStart$sigma
  
  if (pdMat=="pdIdent"){
    parsStart <- covStart$tau
    PsiStart <- parsStart^2*diag(q)
  }
  
  if (pdMat=="pdDiag") {
    parsStart <- rep(covStart$tau, q)
    PsiStart <- diag(parsStart^2, nrow=length(parsStart))
  }
  
  lambdaInvGrp <- mapply(LambdaInv,Z=zGrp,MoreArgs=list(Psi=PsiStart,sigma=sStart))
  fctStart <- ObjFunctionSCAD(xGroup=xGrp,yGroup=yGrp,LGroup=lambdaInvGrp,b=betaStart,weights=weights,
                          lambda=lambda,SCADa=SCADa,nonpen=nonpen,ll1=ll1)
 

  # --- Coordinate Gradient Descent-iteration ------
  # ---------------------------------------------
  
  # some necessary allocations:
  betaIter <- betaStart
  sIter <- sStart
  parsIter <- parsStart
  PsiIter <- PsiStart
  convPar <- crossprod(betaIter)
  convCov <- crossprod(c(sStart,parsStart))
  fctIter <- convFct <- fctStart
  hessian0 <- rep(0,p)
  mat0 <- matrix(0,ncol=p,nrow=N)
  
  stopped <- FALSE
  doAll <- FALSE
  converged <- 0
  counterIn <- 0
  counter <- 0       # counts the number of outer iterations
  
  while ((maxIter > counter) &  (convPar > tol | convFct > tol | convCov > tol | !doAll ))
  {
    # print(counter)
    if (maxIter==counter+1) {
      cat("maxIter reached","\n") ; converged <- converged + 1} #mark that algo didn't converged
    
    # arguments that change at each iteration
    counter <- counter + 1 ; 
    betaIterOld <- betaIter
    fctIterOld <- fctIter
    covIterOld <- c(sIter,parsIter)
    
    
    # --- optimization w.r.t the fixed effects vector beta ---
    # --------------------------------------------------------
    
    activeSet <- which(betaIter!=0)

    # calculate the hessian matrices for j in the activeSet
    HessIter <- HessIterTrunc <- HessianMatrix(xGroup=xGrp,LGroup=lambdaInvGrp,activeSet=activeSet,
                                               N=N,hessian=hessian0,mat=mat0[,activeSet,drop=FALSE])
    HessIter[activeSet] <- pmin(pmax(HessIter[activeSet],lower),upper)
    
    
    fs <- function(x,l,a) {l%*%x[,a]}
    LxGrp <- mapply(fs,xGrp,lambdaInvGrp,MoreArgs=list(a=activeSet),SIMPLIFY=FALSE)
    ll2 <- sum(mapply(nlogdetfun,lambdaInvGrp))

    as3 <- function(s,r,j) crossprod(r,s[,j])
    for (j in activeSet)
    {
        r <- y-x[,-c(j),drop=FALSE]%*%betaIter[-j]
        rGroup <- split(r,grp)
        cut1 <- sum(mapply(as3,LxGrp,rGroup,MoreArgs=list(j=match(j,activeSet))))
        JinNonpen <- j%in%nonpen
      
      # optimum can be calculated analytically
      if (HessIterTrunc[j]==HessIter[j])
      {
        if (JinNonpen) 
          {betaIter[j] <- cut1/HessIter[j]} 
        else 
          {             ##depends on penalty
           if (abs(cut1)<=2*lambda/weights[j]) 
              betaIter[j] <- sign(cut1)*max(abs(cut1)-lambda/weights[j],0)/HessIter[j]
           else if (abs(cut1)>SCADa*lambda/weights[j])
              betaIter[j] <- cut1/HessIter[j]
           else 
              betaIter[j] <- ((SCADa-1)*cut1-sign(cut1)*SCADa*lambda/weights[j])/((SCADa-2)*HessIter[j])
           }
      } 
      else
      {# optimimum is determined by the armijo rule
        armijo <- ArmijoRuleSCAD(xGroup=xGrp,yGroup=yGrp,LGroup=lambdaInvGrp,b=betaIter,j=j,cut=cut1,HkOldJ=HessIterTrunc[j],
                             HkJ=HessIter[j],JinNonpen=JinNonpen,lambda=lambda,SCADa=SCADa,weights=weights,nonpen=nonpen,
                             ll1=ll1,ll2=ll2,converged=converged,
                             control=list(max.armijo=maxArmijo,a_init=a_init,delta=delta,rho=rho,gamma=gamma))
        betaIter <- armijo$b
        converged <- armijo$converged
        fctIter <- armijo$fct
      }
    } 
    
  
    # --- optimization w.r.t the variance components parameters ---
    # -------------------------------------------------------------
    
# calculations before the covariance optimization
    activeset <- which(betaIter!=0)
    rIter <- y-x[,activeset,drop=FALSE]%*%betaIter[activeset,drop=FALSE]
    resGrp <- split(rIter,grp)
    
    ll4 <- OSCAD(betaIter[-nonpen],lambda/weights[-nonpen],SCADa)                 ##depends on penalty
    loglik <- MLloglik(xGroup=xGrp,yGroup=yGrp,LGroup=lambdaInvGrp,b=betaIter,ntot=ntot,N=N,activeSet=which(betaIter!=0))
    print(glue("Cost after updating fixed effects is {ll4 - loglik}"))


# optimization for theta with nlminb (starting value provided)
    if (pdMat=="pdIdent"){
      true.pars=parsIter
      optRes <- nlminb(true.pars,MLpdIdentFct,zGroup=zGrp,resGroup=resGrp, sigma=sIter,LPsi=diag(q),lower = 10^(-6), upper = 10)
      if (optRes$par<thres)
        pars <- 0
      else
        pars <- optRes$par

      fct <- ll1 + optRes$objective + ll4
    
      PsiIter <- pars^2*diag(q)    
    }
    if (pdMat=="pdDiag"){
      true.pars=parsIter
      pars=parsIter
      for (s in 1:q){
        optRes <- nlminb(true.pars[s],MLpdSymFct,zGroup=zGrp,resGroup=resGrp, sigma=sIter,
                         a=s,b=s,LPsi=diag(true.pars, nrow=length(true.pars)),lower = 10^(-6), upper = 10)
      if (optRes$par<thres)
        pars[s] <- 0
      else
        pars[s] <- optRes$par
      
      fct <- ll1 + optRes$objective + ll4
      }
      PsiIter <- diag(pars^2, nrow=length(pars))     
    }
    
    parsIter <- pars
    fctIter <- fct

# optimization of the error variance \sigma^2 with nlminb (starting value provided)
    true.sigma=sIter
    optRes <- nlminb(true.sigma,MLsigmaFct,zGroup=zGrp,resGroup=resGrp,Psi=PsiIter,lower = 10^(-6), upper = 10)
    sIter <- optRes$par
    fctIter <- ll1 + optRes$objective + ll4
    print(glue("Cost after iteration {counter} is {fctIter}"))
    

    lambdaInvGrp <- mapply(LambdaInv,Z=zGrp,MoreArgs=list(Psi=PsiIter,sigma=sIter))
    covIter <- c(sIter,parsIter)
    
    # --- check convergence ---
    convPar <- sqrt(crossprod(betaIter-betaIterOld))/(1+sqrt(crossprod(betaIter)))
    convFct <- abs((fctIterOld-fctIter)/(1+abs(fctIter)))
    convCov <- sqrt(crossprod(covIter-covIterOld))/(1+sqrt(crossprod(covIter)))

    if ((convPar <= tol) & (convFct <= tol) & (convCov <= tol)) counterIn <- 0
    
  }
  
# --- prediction of the random effects ---
# ----------------------------------------
  
biGroup <- u <- list() ; length(biGroup) <- length(u) <- N
Psi <- PsiIter
corPsi <- cov2cor(Psi)
  
  if (pdMat=="pdIdent") cholPsi <- parsIter*diag(q)
  if (pdMat=="pdDiag")  cholPsi <- diag(parsIter, nrow=length(parsIter))
    
  for (i in 1:N)
  {
    u[[i]] <- sIter*solve(t(zGrp[[i]]%*%cholPsi)%*%zGrp[[i]]%*%cholPsi+sIter^2*diag(q))%*%t(zGrp[[i]]%*%cholPsi)%*%resGrp[[i]]
    biGroup[[i]] <- 1/sIter*cholPsi%*%u[[i]]
  }
  
  
  #  --- Some final calculations ---
  # ---------------------------
  
  # fitted values and residuals
  residGrp <- fittedGrp <- list() ; length(residGrp) <- length(fittedGrp) <- N
  for (i in 1:N)
  {
    fittedGrp[[i]] <- xGrp[[i]][,activeSet,drop=FALSE]%*%betaIter[activeSet,drop=FALSE] + zGrp[[i]]%*%biGroup[[i]]
    residGrp[[i]] <- yGrp[[i]]-fittedGrp[[i]]
  }
  residuals <- unlist(residGrp)
  fitted <- unlist(fittedGrp)
  
  # random effects, sorted per subject
  u <- unlist(u) # corresponds to lmer@u
  bi <- unlist(biGroup) # unsorted random effects
  
  # random effects, sorted per effect
  ranef <- bi[order(rep(1:q,N))] # corresponds to lmer@ranef
  
  # fixed effects without names
  fixef <- betaIter
  names(fixef) <- NULL
  
  # --- summary information ---
  # ---------------------------
  npar <- sum(betaIter!=0) + length(c(sIter,parsIter))
  logLik <- MLloglik(xGroup=xGrp,yGroup=yGrp,LGroup=lambdaInvGrp,b=betaIter,ntot=ntot,N=N,activeSet=which(betaIter!=0))
  deviance <- -2*logLik
  aic <- -2* logLik + 2*npar
  bic <- -2* logLik + log(ntot)*npar
  
  if (any(parsIter==0)) cat("Redundant covariance parameters.","\n")
  if (converged>0) cat("Algorithm does not properly converge.","\n")
  if (stopped) {cat("|activeSet|>=min(p,ntot): Increase lambda or set stopSat=FALSE.","\n"); 
  sIter <- parsIter <- nlogLik <- aic <- bic <- NA ; betaIter <- rep(NA,p) ; bi <- fitted <- residuals <- NULL}
  
  out <- list(data=list(x=x,y=y,z=z,grp=grp),weights=weights,coefInit=list(betaStart=betaStart,parsStart=parsStart,sStart=sStart),
              lambda=lambda,sigma=abs(sIter),pars=parsIter,coefficients=betaIter,random=bi,u=u,ranef=ranef,fixef=fixef,fitted.values=fitted,
              residuals=residuals,Psi=Psi,corPsi=corPsi,converged=converged,logLik=logLik,npar=npar,deviance=deviance,
              aic=aic,bic=bic,nonpen=nonpen,counter=counter,call=match.call(),
              stopped=stopped,ranInd=ranInd,objective=fctIter)
  
  out
}

set.seed=532;
data <- read.csv("R/lmmSCAD/random3_covdiag/data47.csv")
grp = data$group
grp = factor(grp)
x = as.matrix(data[,!(colnames(data) %in% c("group","y"))])
z = as.matrix(data[, 2:4])
lambda = 15
y = data[,ncol(data)]
results = lmmSCAD(x, y, z, grp, lambda = lambda, pdMat = "pdDiag")
results$coefficients[1:10]
sum(results$coefficients != 0)
results$pars
results$coefInit$betaStart[1:10]
sum(results$coefInit$betaStart != 0)
# βnz2 = [1,2,4,3,3,-1,5,-3,2,2]


