
##################### -- generate an Identity matrix of order as nrow(z)

ZIdentity <- function(Z)
{
  ZId <- diag(dim(Z)[[1]])
  return(list(ZId))
}

######################-- Return Lambda & Lambda^{-1} where Lambda = sigma^2*ZId + z^T Psi Z
MLpdSymLambda <- function(Z,Psi,sigma)
{
  ZId <- diag(dim(Z)[[1]])
  Lambda <- sigma^2*ZId + quad.tform(Psi,Z)
  return(list(Lambda))
}

LambdaInv <- function(Z,Psi,sigma)
{
  ZId <- diag(dim(Z)[[1]])
  LambdaInverse <- solve(sigma^2*ZId + quad.tform(Psi,Z)) 
  return(list(LambdaInverse))
}


##################### -- calculate objective function
nlogdetfun <- function(L){ -1/2*determinant(L)$modulus[1]}


ObjFunctionSCAD <- function(xGroup,yGroup,LGroup,b,weights,lambda,SCADa,nonpen,ll1=ll1)
{
  ResAs <- function(x,y,b,activeSet) y-x[,activeSet,drop=FALSE]%*%b[activeSet,drop=FALSE]
  tResLRes <- function(L,res) 1/2*quad.form(L,res) 
  

  activeSet <- which(b!=0)
  
  ll2 <- sum(mapply(nlogdetfun,LGroup))
  resGroup <- mapply(ResAs,x=xGroup,y=yGroup,MoreArgs=list(b=b,activeSet=activeSet),SIMPLIFY=FALSE)
  ll3<-sum(mapply(tResLRes,LGroup,resGroup))
  lp<-  OSCAD(b[-nonpen],lambda/weights[-nonpen],SCADa)   ##depends on penalty
  
  ll <- ll1 + ll2 + ll3 + lp
  return(Fc=ll)      
}

##################### -- calculate Hessian Matrix

HessianMatrix <- function(xGroup,LGroup,activeSet,N,hessian,mat)
{
  for (i in 1:N)
  {  
    mat[i,] <- diag(t(xGroup[[i]][,activeSet,drop=FALSE])%*%LGroup[[i]]%*%xGroup[[i]][,activeSet,drop=FALSE])
  }
  hessian[activeSet] <- apply(mat,2,sum)
  return(hessian)
}



##################### -- calculate beta_j update by Armijio rule
ArmijoRuleSCAD <- function(xGroup,yGroup,LGroup,b,j,cut,HkOldJ,HkJ,JinNonpen,lambda,SCADa,
                           weights,nonpen,ll1,ll2,converged,control)
{
  b.new <- b
  bJ <- b[j]
  grad <- -cut + bJ*HkOldJ
  if (JinNonpen) 
    {dk <- -grad/HkJ} 
  else 
    {   ##depends on penalty
      ck<- (bJ*HkJ)-grad
      lambdaw<-lambda/weights[j]
      if (ck<=lambdaw*(Hkj+1))
        dk<- (-lambdaw-grad)/HkJ
      else if (ck>lambdaw*SCADa*HkJ)
        dk<- -grad/HkJ
      else
        dk <- (-grad*(SCADa-1)-(SCADa*lambdaw - bJ))/(Hkj*(SCADa-1)-1)
      }
  
  if (dk!=0)
  { 
    # calculate delta_k
    if (JinNonpen) 
      deltak <- dk*grad + control$gamma*dk^2*HkJ
    else {   ##depends on penalty
  deltak <- dk*grad + control$gamma*dk^2*HkJ + OSCAD(bJ+dk,lambda/weights[j],SCADa)-OSCAD(bJ,lambda/weights[j],SCADa)
      }
    
    
    fctOld <- ObjFunctionSCAD(xGroup=xGroup,yGroup=yGroup,LGroup=LGroup,b=b,weights=weights,
                          lambda=lambda, SCADa, nonpen=nonpen,ll1=ll1)
    for (l in 0:control$max.armijo)
    { 
      b.new[j] <- bJ + control$a_init*control$delta^l*dk
      
      fctNew <- ObjFunctionSCAD(xGroup=xGroup,yGroup=yGroup,LGroup=LGroup,b=b.new,weights=weights,
                            lambda=lambda, SCADa, nonpen=nonpen,ll1=ll1)
      addDelta <- control$a_init*control$delta^l*control$rho*deltak
      if (fctNew <= fctOld + addDelta)
      {
        b[j] <- bJ + control$a_init*control$delta^l*dk
        fct <- fctNew
        break
      }
      if (l==control$max.armijo)
      {
        if (trace>2) cat("Armijo for b_",j," not successful","\n")
        converged <- converged + 2
        fct <- fctOld
      }
    }
  } 
  return(list(b=b,fct=fct,converged=converged))
}


##################### -- Obj function for calculating ML of theta in Psi (depends on Psi = theta^2 Identity)
MLpdIdentFct <- function(thetak,zGroup,resGroup,sigma,LPsi)
{
  Psi <- thetak^2*LPsi
  LambdaGroup <- mapply(MLpdSymLambda,Z=zGroup,MoreArgs=list(sigma=sigma,Psi=Psi))
  
  ll2 <- mapply(MLpdSymObj,LambdaGroup,resGroup)
  1/2*sum(ll2)
}

MLpdSymObj <- function(Lambda,res)  determinant(Lambda)$modulus[1] + crossprod(res, solve(Lambda, res))


##################### -- Obj function for calculating ML of theta's in diag Psi (depends on Psi = Diag(theta^2))
MLpdSymFct <- function(thetak,zGroup,resGroup,sigma,a,b,LPsi)
{
  LPsi[a,b] <- thetak
  Psi <- tcrossprod(LPsi)
  LambdaGroup <- mapply(MLpdSymLambda,Z=zGroup,MoreArgs=list(sigma=sigma,Psi=Psi))
  
  ll2 <- mapply(MLpdSymObj,LambdaGroup,resGroup)
  1/2*sum(ll2)
  
}


##################### -- Obj function for calculating ML of sigma^2  ----------

MLsigmaFct <- function(sigma,zGroup,resGroup, Psi)
{
  LambdaGroup <- mapply(MLpdSymLambda,Z=zGroup,MoreArgs=list(sigma=sigma,Psi=Psi))

  ll2 <- mapply(MLpdSymObj,LambdaGroup,resGroup)
  1/2*sum(ll2)
}



#################### -- ML log-likelihod ----------

MLloglik <- function(xGroup,yGroup,LGroup,b,ntot,N,activeSet)
{
  
  l1 <- l2 <- numeric(N) ; ll2b <- 0
  for (i in 1:N)
  {
    l1[i] <- -determinant(LGroup[[i]])$modulus
    l2[i] <- quad.form(LGroup[[i]],yGroup[[i]]-xGroup[[i]]%*%b)
  }
  
  ll <- - 1/2*(sum(l1) + sum(l2) + ntot*log(2*pi) + ll2b)
  return(loglik=ll)    
}




#################### -- Starting value of the covariates as in Schelldorfer et al. (2011) paper ----------


covStartingValues <- function(xGroup,yGroup,zGroup,zIdGroup,b,ntot,N,lower=-10,upper=10)
{
  optimize1 <- function(x,y,b) y-x%*%b
  
  optimize2 <- function(gamma,zId,ZtZ)
  {
    H <- zId + exp(2*gamma)*ZtZ
    return(list(H=H))
  }
  
  optimize3 <- function(res,zId,ZtZ,gamma)
  {
    lambda <- optimize2(gamma,zId,ZtZ)
    logdetH <- determinant(lambda$H)$modulus
    quadH <- quad.form.inv(lambda$H,res)
    return(c(logdetH,quadH))
  }
  
  optimize4 <- function(gamma)
  {
    optH <- mapply(optimize3,resGroup,zIdGroup,ZtZ=ztzGroup,MoreArgs=list(gamma=gamma))
    H1 <- optH[1,]
    H2 <- optH[2,]
    
    fn <- ntot*log(sum(H2)) + sum(H1)
    fn
  }
  
  optimize5 <- function(z) tcrossprod(z)
  
  resGroup <- mapply(optimize1,x=xGroup,y=yGroup,MoreArgs=list(b=b),SIMPLIFY=FALSE)
  ztzGroup <- mapply(optimize5,z=zGroup,SIMPLIFY=FALSE)
  
  optRes <- optimize(f=optimize4,interval=c(lower,upper))
  
  gamma <- optRes$minimum
  
  quadH <- mapply(optimize3,resGroup,zIdGroup,ztzGroup,MoreArgs=list(gamma=gamma))[2,]
  
  sig <- sqrt(1/ntot*sum(quadH))
  tau <- exp(gamma)*sig
  objfct <- 1/2*(optRes$objective + ntot*(1-log(ntot)))
  
  return(list(tau=tau,sigma=sig,opt=objfct))
}



#################### -- SCAD Penalty without derivative ----------

OSCAD <- function(beta, lambda, a)
{
  alpha=-(2*a-2)^(-1)
  u=abs(beta)
  f=beta
  for (i in 1:length(beta))
  {
  b=a*lambda[i]/(a-1)
  c=lambda[i]^2/(2-2*a)
  if (u[i]<lambda[i]) {
    f[i]=lambda[i]*u[i]
  }   
  else 
    {
    if (u[i]<a*lambda[i]) {
      f[i]=alpha*u[i]^2+b*u[i]+c
    }   
    else {
      f[i]=(1+a)*lambda[i]^2/2
    }   
    }
  }
  return(sum(f))
}