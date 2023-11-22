library(ggmix)

##INFORMATION ABOUT gen_structured_model function

## ARGUMENTS: 
# n number of observations to simulate
# p_design number of variables in X_test, i.e., the design matrix
# p_kinship number of variable in X_kinship, i.e., matrix used to calculate kinship
# k number of intermediate subpopulations.
# s the desired bias coefficient, which specifies sigma indirectly. Required if sigma is missing
# Fst The desired final FST of the admixed individuals. Required if sigma is missing
# b0 the true intercept parameter
# nPC number of principal components to include in the design matrix used for regres- sion adjustment for population structure via principal components. This matrix is used as the input in a standard lasso regression routine, where there are no random effects.
# eta the true eta parameter, which has to be 0 < eta < 1 sigma2 the true sigma2 parameter
# geography the type of geography for simulation the kinship matrix. "ind" is independent
# populations where every individuals is actually unadmixed, "1d" is a 1D geography "circ" is circular geography. Default: "ind". See the functions in thebnpsd for details on how this data is actually generated.
# percent_causal percentage of p_design that is causal. must be 0 < percentcausal <1. The true regression coefficients are generated from a standard normal distribution.
# percent_overlap this represents the percentage of causal SNPs that will also be included in the calculation of the kinship matrix
# train_tune_test the proportion of sample size used for training tuning parameter selection and testing. default is 60/20/20 split


##OUTPUT
# ytrain simulated response vector for training set
# ytune simulated response vector for tuning parameter selection set
# ytest simulated response vector for test set
# xtrain simulated design matrix for training set
# xtune simulated design matrix for tuning parameter selection set
# xtest simulated design matrix for testing set
# xtrain_lasso simulated design matrix for training set for lasso model. This is the same as xtrain, but also includes the nPC principal components
# xtune_lasso simulated design matrix for tuning parameter selection set for lasso model. This is the same as xtune, but also includes the nPC principal components
# xtest simulated design matrix for testing set for lasso model. This is the same as xtest, but also includes the nPC principal components
# causal character vector of the names of the causal SNPs
# beta the vector of true regression coefficients
# kin_train 2 times the estimated kinship for the training set individuals
# kin_tune_train The covariance matrix between the tuning set and the training set individuals
# kin_test_train The covariance matrix between the test set and training set individuals
# Xkinship the matrix of SNPs used to estimate the kinship matrix
# not_causal character vector of the non-causal SNPs
# PC the principal components for population structure adjustment

set.seed(123)
for (i in 1:100) {
    admixed <- gen_structured_model(n = 250,
                                p_design = 1000,
                                p_kinship = 7e2,
                                geography = "1d",
                                percent_causal = 0.10,
                                percent_overlap = "100",
                                k = 10, s = 0.5, Fst = 0.1,
                                b0 = 0, nPC = 10,
                                eta = 0.1, sigma2 = 1,
                                train_tune_test = c(0.9, 0.05, 0.05))
    X <- rbind(admixed$xtrain, admixed$xtune, admixed$xtest)
    write.csv(X, paste0("data/GWAS/data", i, ".csv"), row.names = FALSE)
}

