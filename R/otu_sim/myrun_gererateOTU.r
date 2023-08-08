##############################################package
library(penalized)
library(emulator)
library(MASS)
library(qpcR)
library(iZID)
source('sim_OTU.R')
##################################################################
set.seed(144)
N <- 25           # number of groups
p <- 200            # number of covariates (including intercept)
q <- 2          # number of random effect covariates ##q first column of x
ni <- rep(6,N) # observations per group
sigma=0.25
theta1=0.56
theta=rep(theta1, q)
beta <- c(1,c(2,4,3,3,0),rep(0,p-5)) # fixed-effects coefficients 
dis="bnb"
par_dis =c(10,8,9)
phi=0.6
data1 = sim_OTU(N,p,q,ni,beta,sigma,theta,dis,par_dis,phi)
str(data1)
##########################################################################################
library(dirmult)
A = simPop(J=2, K=10,n=20, theta=0.2)
AA = A$data
sum(AA==0)
simPop(J=10, K=5,n=10, theta=0.03)











######################################Ezafiiii
######################################################################
set.seed(123)  # set the seed for reproducibility
# generate the matrix
mat <- matrix(0, n, r)  # initialize a matrix of zeros
for (i in 1:r) {
  mat[, i] <- sample.zi(n,phi=0.6, distri = dis,r=10,alpha1=8,alpha2=9)
}
# view the matrix
mat
sum(mat==0)




library(iZID)
set.seed(123)  # set the seed for reproducibility

n <- 10  # number of rows
dis="bnb"  # distribution of the random variable
r <- 10  # number of columns
alpha1 <- 8  # value of alpha1 parameter
alpha2 <- 9  # value of alpha2 parameter

# generate the matrix
mat <- matrix(0, n, r)  # initialize a matrix of zeros
for (i in 1:r) {
  mat[, i] <- sample.h(n,phi=0.6, distri = dis,r=10,alpha1=8,alpha2=9)
}
# view the matrix
mat
sum(mat==0)












