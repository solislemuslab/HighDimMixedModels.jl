using Revise
using HighDimMixedModels
using RCall
using DataFrames
using StatsBase
using Random
using MLBase
using Lasso
using JLD2

R"""
library(BGLR)
data(mice)

#Create penalized high dimensional genotype design matrix G
G = mice.X
snp_names = colnames(G)

# Create unpenalized low dimensional design matrix X
pheno = mice.pheno
# Age missing some values--replace with median
age = pheno$Biochem.Age
cat("Number of missing age observations: ", sum(is.na(age)), "\n")

pheno[is.na(age), "Biochem.Age"] = median(age, na.rm = TRUE)
X = model.matrix( ~ GENDER +  Biochem.Age + factor(Litter), data=pheno)

#Cage group structure for random effects 
grp = mice.pheno$cage

#Response variable
y = mice.pheno$Obesity.BMI
"""

@rget G
@rget snp_names
@rget X
@rget y
@rget grp
grp = String.(grp)
N = length(y)
p = size(G, 2)
q = size(X, 2)
XG = [X G]
Z = ones(N,1) # just random intercept 
y_cent = (y .- mean(y))./std(y) # center and scale response to match the model fit with BGLR

control = Control()
control.trace = 3
control.tol = 1e-2
Random.seed!(1234)

gwas_fit1 = lmmlasso(X, G, y_cent, grp, Z; standardize = true,
    penalty="scad", λ=150, ψstr="ident", control=control)

gwas_fit2 = lmmlasso(X, G, y_cent, grp, Z; standardize = true,
    penalty="scad", λ=190, ψstr="ident", control=control)

gwas_fit3 = lmmlasso(X, G, y_cent, grp, Z; standardize = true,
    penalty="scad", λ=200, ψstr="ident", control=control)

save_object("data/real/GWAS/gwas_fit200.jld2", gwas_fit)
save_object("data/real/GWAS/gwas_fit150.jld2", gwas_fit2)
save_object("data/real/GWAS/gwas_fit190.jld2", gwas_fit3)


################
gwas_fit1 = load_object("data/real/GWAS/gwas_fit150.jld2")
gwas_fit2 = load_object("data/real/GWAS/gwas_fit190.jld2")
gwas_fit3 = load_object("data/real/GWAS/gwas_fit200.jld2")
R"load(\"R/real_gwas_data/fm.rda\")" # this model was fit in the file R/real_gwas.R

snp_coef1 = gwas_fit1.fixef[11:end]
idx_snp_nz1 = snp_coef1 .!= 0;
snp_names[idx_snp_nz1]

snp_coef2 = gwas_fit2.fixef[11:end]
idx_snp_nz2 = snp_coef2 .!= 0;
snp_names[idx_snp_nz2]

snp_coef3 = gwas_fit3.fixef[11:end]
idx_snp_nz3 = snp_coef3 .!= 0;
snp_names[idx_snp_nz3]

R"snp_coef_bglr = fm$ETA$MRK$b"

#Coefficient dot plot
R"""
library(tidyverse)
library(patchwork)
theme_set(theme_bw())
theme_update(panel.grid = element_blank())
plot_snp_effects <- function(snp_effects, title) {
    snp_df = data.frame(coef = snp_effects, abs_coef = abs(snp_effects), marker = 1:length(snp_effects))
    p = ggplot(snp_df) +
        geom_segment(aes(x = marker, xend = marker, y = 0, yend = abs_coef), color = "red") +
        geom_point(aes(x = marker, y = abs_coef), color = "blue") +
        labs(y = "coefficient maginiude") +
        theme(text = element_text(size = 15))

    fname = paste0("plots/real/",title,".pdf")
    ggsave(fname,p,width = 20, height = 15, units = "cm")
    return(p)
}

p4 = plot_snp_effects(snp_coef_bglr, "snp_effects_bglr")

snp_coef1 = $snp_coef1
snp_coef2 = $snp_coef2
snp_coef3 = $snp_coef3
p1 = plot_snp_effects(snp_coef1, "snp_effects_150")
p2 = plot_snp_effects(snp_coef2, "snp_effects_190")
p3 = plot_snp_effects(snp_coef3, "snp_effects_200")
patchwork = (p3 + p2)/(p1+p4) +  plot_annotation(tag_levels = 'A')
ggsave("plots/real/all_snp_effects.pdf", patchwork, width=40, height=30, units="cm")
"""


#True versus predicted plots
R"predicts_bglr_fixed = fm$mu + X[,-1]%*%fm$ETA$FIXED$b + G%*%snp_coef_bglr"
R"predicts_bglr = fm$yHat"
predicts3_fixed = X*gwas_fit3.fixef[1:10] + G*gwas_fit3.fixef[11:end]
predicts3 = similar(y)
for (i, group) in enumerate(unique(grp))
    predicts3[grp.==group] = gwas_fit3.fitted[i]
end
R"""
theme_set(theme_bw())
predicts3 = $predicts3
predicts3_fixed = $predicts3_fixed
y_cent = $y_cent
# Plot true phenotype versus predicted from BGLR without random effects
pheno_df = data.frame(true = y_cent, predicted = predicts_bglr_fixed, gender = mice.pheno$GENDER)
p1 <- ggplot(pheno_df) +
    geom_point(aes(x = predicted, y = true, color = gender)) +
    geom_abline(intercept = 0, slope = 1, color = "blue") +
    theme_minimal() +
    ggtitle("BGLR without \nrandom effects")
# Plot true phenotype versus predicted from BGLR with random effects
pheno_df$predicted = predicts_bglr
p2 <- ggplot(pheno_df) +
    geom_point(aes(x = predicted, y = true, color = gender)) +
    geom_abline(intercept = 0, slope = 1, color = "blue") +
    theme_minimal() +
    ggtitle("BGLR with \nrandom effects")
# Plot true phenotype versus predicted from our model without random effects 
pheno_df$predicted = predicts3_fixed
p3 <- ggplot(pheno_df) +
    geom_point(aes(x = predicted, y = true, color = gender)) +
    geom_abline(intercept = 0, slope = 1, color = "blue") +
    theme_minimal()+
    ggtitle("Our model without \nrandom effects")
# Plot true phenotype versus predicted from our model with random effects
pheno_df$predicted = predicts3
p4 <- ggplot(pheno_df) +
    geom_point(aes(x = predicted, y = true, color = gender)) +
    geom_abline(intercept = 0, slope = 1, color = "blue") +
    theme_minimal()+
    ggtitle("Our model with \nrandom effects")

(p1 + p2)/(p3+p4) + plot_layout(guides = "collect")
ggsave("plots/real/gwas_true_v_predicted.pdf", width = 25, height = 25, units = "cm")
"""


#Fit and variance statistics
predicts_bglr_fixed = @rget predicts_bglr_fixed
predicts_bglr = @rget predicts_bglr
var(y_cent-predicts_bglr)
var(y_ce)
var(y_cent - predicts3_fixed)
var(y_cent-predicts3)

gwas_fit3.σ²
R"fm$varE"

gwas_fit3.L^2
R"fm$ETA$CAGE$varB"

#names of SNPs with non-zero coefficients
my_names = snp_names[idx_snp_nz3]
R"""
my_names = $my_names
high_effect = quantile(abs(snp_coef_bglr), (length(snp_coef_bglr) - 10)/length(snp_coef_bglr))
their_names = snp_names[abs(snp_coef_bglr) > high_effect]
intersect(their_names, my_names)
"""
# all seven of our non-zero SNPs are in the top 20 SNPs in the BGLR model


