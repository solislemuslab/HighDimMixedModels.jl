library(tidyverse)
library(splmm)

##### Riboflavin data ####
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








