library(tidyverse)
library(splmm)

##### Real genetic (Riboflavin) data ####
data <- read_csv("data/real/gene_expressions/riboflavingrouped.csv")
ribo <- as_tibble(t(data[,-1]))
colnames(ribo) <- data[[1]]

grp <- read_csv("data/real/gene_expressions/riboflavingrouped_structure.csv", 
                col_names = F) 
grp <- factor(grp$X1)
N = nrow(ribo)
X <- cbind(rep(1, N), as.matrix(ribo[,-1]))
Z <- as.matrix(rep(1, N))
y <- ribo$q_RIBFLV
control = splmmControl()
control$trace = 3
lasso_fit <- splmm(X, y, Z, grp, 
                   lam1=2, lam2=1, penalty.b="lasso", control = control)



##### Log-ratio normalized OTU data #######
# First column ("group") is group identity 
# Second column ("X1") is all ones
# Third column ("X2") is variable measured at group level
# Fourth through second to last column are regular OTU predictors (reference OTU column has already been removed)
# Last column is the response, which is generated with a random intercept
otu_data <- read_csv("data/OTU/random1_covid/data_cluster_1.csv")

group <- factor(otu_data$group)

X <- otu_data %>%
  select(-c(group, y)) %>%
  as.matrix()

Z <- otu_data %>%
  select(X1) %>%
  as.matrix()

y <- otu_data$y

system.time(lasso_fit <- splmm(X, y, Z, group, lam1=2, lam2=1, nonpen.b=1:2, penalty.b="lasso"))
summary(lasso_fit)
lasso_fit$coefInit[[1]]

scad_fit = splmm(X, y, Z, group, lam1 = .1, lam2 = 1, nonpen.b=1:2, penalty.b = "scad")
summary(scad_fit)
scad_fit$coefInit[[1]]
scad_fit$converged
# Scad fit is very off (even though both fits start with the same initial estimate)

# Same type of data as above but with different generation of OTU count matrix 
otu_data <- read_csv("data/OTU/random1_covid/data_erdos_renyi_1.csv")

group <- factor(otu_data$group)

X <- otu_data %>%
  select(-c(group, y)) %>%
  as.matrix()

Z <- otu_data %>%
  select(X1) %>%
  as.matrix()

y <- otu_data$y

lasso_fit <- splmm(X, y, Z, group, lam1 = 2, lam2 = 1, nonpen.b=1:2, penalty.b = "lasso")
summary(lasso_fit)
lasso_fit$coefInit

scad_fit = splmm(X, y, Z, group, lam1 = .1, lam2 = 1, nonpen.b=1:2, penalty.b = "scad")
summary(scad_fit)
scad_fit$converged
#scad fit is very off and in addition converged does not reflect the fact that maximum # of iterations was reached

# Now we use data with three random effects (but identity covariance struct)
otu_data <- read_csv("data/OTU/random3_covid/data_erdos_renyi_1.csv")

group <- factor(otu_data$group)

X <- otu_data %>%
  select(-c(group, y)) %>%
  as.matrix()

Z <- otu_data %>%
  select(X1:X3) %>%
  as.matrix()

y <- otu_data$y

lasso_fit <- splmm(X, y, Z, group, lam1=1.6, lam2=1, 
                   nonpen.b=1:4, nonpen.L=1:3, penalty.b="lasso")
summary(lasso_fit)
lasso_fit$coefInit

scad_fit = splmm(X, y, Z, group, lam1 = .1, lam2 = 1, 
                 nonpen.b=1:4, nonpen.L=1:3, penalty.b = "scad")
summary(scad_fit)
scad_fit$converged


##### Simulated GWAS data with ~1000 columns ######
gwas_data <- read_csv("data/GWAS/random1_covid/data1.csv")

group <- factor(gwas_data$group)

X <- gwas_data %>%
  select(-c(group, y)) %>%
  as.matrix()

Z <- gwas_data %>%
  select(X1) %>%
  as.matrix()

y <- gwas_data$y

system.time(lasso_fit <- splmm(X, y, Z, group, lam1=5, lam2=1, penalty.b="lasso"))
summary(lasso_fit)
head(lasso_fit$coefInit[[1]], 10)
sum(lasso_fit$coefInit[[1]] != 0)
sum(lasso_fit$coefficients != 0)

system.time(scad_fit <- splmm(X, y, Z, group, lam1=5, lam2=1, penalty.b="scad"))
summary(scad_fit)
head(scad_fit$coefInit[[1]], 10)
sum(scad_fit$coefInit[[1]] != 0)
sum(scad_fit$coefficients != 0)

tp = c(1,2,4,3,3,-1,5,-3,2,2,0)
cbind(tp, init = lasso_fit$coefInit[[1]][1:11], 
      scad = scad_fit$coefficients[1:11], lasso = lasso_fit$coefficients[1:11])
c("lasso_aic" = lasso_fit$aic, "scad_aic" = scad_fit$aic)















