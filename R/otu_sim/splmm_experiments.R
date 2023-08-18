library(tidyverse)
library(splmm)

# Cluster, compositional
data_cl10 <- read_csv("data/OTU/random1_covid/data_cluster_30.csv")

group <- data_cl10$group

X <- data_cl10 %>%
  select(-c(1, ncol(data_cl10))) %>%
  as.matrix()

#Get rid of last column to deal with compositional collinearity
X <- X[,-ncol(X)] 

Z <- data_cl10 %>%
  select(X1) %>%
  as.matrix()

y <- data_cl10$y

summary(spfit1 <- splmm(X, y, Z, group, lam1 = 3, lam2 = 1, 
                        nonpen.b = 1, nonpen.L = 1, penalty.b = "lasso"))

# Erdos Renyi, non-compositional
data_er10 <- read_csv("data/OTU/random3_covid/data_erdos_renyi_80.csv")

group <- data_er10$group

X <- data_er10 %>%
  select(-c(1, ncol(data_er10))) %>%
  as.matrix()

Z <- data_er10 %>%
  select(X1:X3) %>%
  as.matrix()

y <- data_er10$y

summary(spfit1 <- splmm(X, y, Z, group, lam1 = .005, lam2 = 100,
                        nonpen.b = 1:3, nonpen.L = 1:3, penalty.b = "scad"))






