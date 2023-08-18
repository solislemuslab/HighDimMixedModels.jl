# Simulate microbiome data according to (Lin, Shi, Feng, Li 2014)

one_design <- function(n = 120, p = 200, theta = NULL, rho = .2, log_ratio = TRUE) {
  
  #Generate vector of means for predictors. Choose first 5 OTUs to be abundant
  if (missing(theta)) theta = c(rep(log(p/2), times = 5), rep(0, times = p-5)) 
  
  #Generate covariance matrix of predictors
  sigma = matrix(0, p, p)
  for (i in 1:p) {
    for (j in 1:p) {
      sigma[i,j] = rho^(abs(i-j))
    }
  }
  
  #Generate W as a multivariate normal, X as exp(W) (normalized), and Z as log(X)
  W = MASS::mvrnorm(n, theta, sigma)
  X = exp(W) / rowSums(exp(W))
  Z = log(X)
  
  #If we want log-ratios instead of just logs (note, last column will be all 0 but will get removed by Julia script)
  if (log_ratio) {
    ref_otu = Z[,ncol(Z)]
    Z = Z - ref_otu
  }
  
  return(list(X = X, Z = Z))
  
}

nd = 100 #Number of data sets to generate under each setting
set.seed(1000)
for (j in 1:nd) {
  
  des_mat = one_design()
  write.csv(des_mat$X, paste0("data/OTU/LinShi/rs_data", j, ".csv"), row.names = FALSE)
  #write.csv(des_mat$Z, paste0("data/OTU/LinShi/lr_data", j, ".csv"), row.names = FALSE)
  print(j)
  
}








