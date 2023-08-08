rm(list=ls())
library(SpiecEasi)
#library(NetCoMi)
library(SPRING)


##########
###  synth data SE from github SE       https://github.com/zdk123/SpiecEasi
##########
data(amgut1.filt)
depths <- rowSums(amgut1.filt)
amgut1.filt.n  <- t(apply(amgut1.filt, 1, norm_to_total))
amgut1.filt.cs <- round(amgut1.filt.n * min(depths))

d <- ncol(amgut1.filt.cs)
n <- nrow(amgut1.filt.cs)
e <- d # the number of edges is set as the number of nodes.

set.seed(10010)
# available types in SpiecEasi: "band", "cluster", "scale_free", "erdos_renyi", "hub", "block".
graph <- SpiecEasi::make_graph('cluster', d, e) 
Prec  <- graph2prec(graph)  
Cor   <- cov2cor(prec2cov(Prec))

X <- synth_comm_from_counts(amgut1.filt.cs, mar=2, distr='zinegbin', Sigma=Cor, n=n)
num_zeros <- sum(X == 0)
num_zeros/(n*d)
########
########***********************************
##  synth data SP from github SP    https://github.com/GraceYoon/SPRING/blob/master/man/synthData_from_ecdf.Rd
##########################

# goal is to generate synthetic data with a prescribed graph structure.
# load real data "QMP" in SPRING package.
data(QMP)
set.seed(12345) # set the seed number for make_graph part.
p1 = ncol(QMP) # the number of nodes.
e1 = 2*p1 # the number of edges is set as twice the number of nodes.
gtype = "cluster"
# available types in SpiecEasi: "band", "cluster", "scale_free", "erdos_renyi", "hub", "block".
graph_p1 <- SpiecEasi::make_graph(gtype, p1, e1) # adjacency matrix. 1: edge, 0: no edge.
Prec1  <- SpiecEasi::graph2prec(graph_p1) # precision matrix. inverse of covariance.
Cor1   <- cov2cor(SpiecEasi::prec2cov(Prec1)) # correlation matrix.

X1_count <- synthData_from_ecdf(QMP, Sigma = Cor1, n = 100)
# generate data of size n by p.
# p = ncol(Cor1) = ncol(QMP) should hold.
# need to specify sample size n.

num_zeros <- sum(X1_count == 0)
num_zeros/(n*d)
