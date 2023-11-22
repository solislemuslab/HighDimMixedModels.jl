#rm(list=ls())
#library(edgeR)
#library(scran)
library(dplyr)
load("data/real/OTU/ageset.Rdata")

OTU_original <- data.obj$otu.tab
OTU_original <- t(OTU_original)
#filter OTU

# Filter based on OTU prevalence
prevalence_threshold <- 0.1
OTU_prevalence <- apply(OTU_original != 0, 2, mean)
OTU_keep <- OTU_prevalence >= prevalence_threshold

# Filter based on median non-zero counts
median_count_threshold <- 10
median_counts <- apply(OTU_original, 2, function(x) median(x[x != 0], na.rm = TRUE))
OTU_keep <- OTU_keep & median_counts >= median_count_threshold

# Subset the OTU table based on the filtered columns
OTU_filtered <- OTU_original[, OTU_keep]
dim(OTU_filtered)
# calculate 97% quantile for each taxon
quantiles <- apply(OTU_filtered, 2, quantile, probs = 0.97)

# Replace values greater than the 97th percentile with the 97th percentile value
for (i in 1:ncol(OTU_filtered)) {
  OTU_filtered[OTU_filtered[,i] > quantiles[i], i] <- quantiles[i]
}
dim(OTU_filtered)

OTU_filtered = sqrt(OTU_filtered)

OTU_filtered_N<- t(apply(OTU_filtered, 1, function(OTU_filtered) OTU_filtered/sum(OTU_filtered)))


OTU = OTU_filtered_N


meta_data <- data.obj$meta.dat
age <- meta_data['age']
age <-sqrt(age)
age <- data.frame(data_name = rownames(age), age = age$age)
OTU<-data.frame(data_name = rownames(OTU),OTU)
# Merge the data frames
merged_data <- merge(OTU, age, by = "data_name")



country = meta_data['geo_loc_name']
# Create a new column that maps each country name to an integer value
country$geo_loc_name <- ifelse(country$geo_loc_name == "USA", 1, 
                               ifelse(country$geo_loc_name == "Malawi", 2,
                                      ifelse(country$geo_loc_name == "Venezuela", 3, NA)))

country$geo_loc_name = as.factor(country$geo_loc_name)
country <- data.frame(data_name = rownames(country), country = country$geo_loc_name)
merged_data2 <- merge(country,merged_data, by = "data_name")


#merged_data2 <- merged_data2 %>% select(-data_name)
# Delete the column named "data_name"
merged_data2  <- merged_data2 [, !colnames(merged_data2 ) %in% "data_name"]


dim(merged_data2)

merged_data2 <- merged_data2[complete.cases(merged_data2$age), ]
dim(merged_data2)


data = merged_data2

write.csv(data, "data/real/OTU/tss_normalized_data.csv", row.names = F)


############################################Lmmscad
library(stringr)
library(penalized)
library(emulator)
library(MASS)
# Define block sizes
psi_block_size <- 1
random_block_size<- 3
beta_block_size <- dim(data)[2]

l1      <- seq(0,10,0.1)
# Initialize results matrix
n_lambdas <- length(l1)
n_cols <- 7 + psi_block_size + random_block_size+beta_block_size 
results <- matrix(NA, nrow = n_lambdas, ncol = n_cols)
colnames(results) <- c("lambda", "logLik", "AIC", "BIC", "RSS", "n_nonzero_B", "sigma",
                       paste0("psi_", rep(1:psi_block_size)),
                       paste0("random_", rep(1:random_block_size)),
                       paste0("beta_", 1:beta_block_size))

ResultsFinal <-c()
# Set the path to the parent folder
error_folder <- "error"

error_count <- 1

#Var_vector = c("pdIdent","pdDiag","pdSym")
#for (V in 1:3){
Var="pdIdent"
for (i in 1:length(l1)){
  
  grp <- data[,1]
  #grp   <- as.factor(grp )
  y <- as.matrix(data[,ncol(data)])
  x <- as.matrix(data[ , -(c(1,ncol(data)))])
  z <- as.matrix(x[,1])
  
  tryCatch({
    set.seed(144)
    fit1 <- lmmSCAD(x, y, z = z, grp, lambda = l1[i], nonpen = 1:dim(z)[[2]], pdMat = Var, SCADa=3.7)
   
    # Store model information in results matrix
    results[i, 1:7] <- c(l1[i],fit1$logLik, fit1$aic, fit1$bic, sum((y - fit1$fitted.values)^2),
                         length(which(fit1$coefficients != 0)), fit1$sigma)
    # Store psi values in results matrix
    psi_values <- c(diag(fit1$Psi), fit1$Psi[upper.tri(fit1$Psi)])
    psi_n_vals <- length(psi_values)
    
    if (psi_n_vals==1){results[i, 8] <- c(diag(fit1$Psi), fit1$Psi[upper.tri(fit1$Psi)])
    } else {results[i, 8:(8+psi_n_vals-1) ] <- c(diag(fit1$Psi), fit1$Psi[upper.tri(fit1$Psi)])}
    results[i, (8+psi_block_size):(8+psi_block_size+random_block_size-1) ] <-c(as.numeric(fit1$ranef),rep(NA,random_block_size-length(as.numeric(fit1$ranef))))
    results[i, (8+psi_block_size+random_block_size):(8+psi_block_size+random_block_size+beta_block_size-1) ] <-c(as.numeric(fit1$coefficients),rep(NA,beta_block_size-length(as.numeric(fit1$coefficients))))
    
    # Inside the tryCatch() function:
  },error = function(e) {
    # Store the error message in the variable
    error_message <- paste0("Error in ", "/data.csv: ", e$message)
    
    # Print the error message
    message(error_message)
    
    # Generate unique file name for the error message
    file_name <- paste0("error_", error_count, ".txt")
    
    # Save the error message as a text file
    writeLines(error_message, file.path(error_folder, file_name))
    
    # Increment the error counter
    error_count <- error_count + 1
  })
}
ResultsFinal <- results[order(results[, 'BIC'], decreasing = FALSE), ]
#Resultscad = c(Var,Resultscad)
#ResultsFinal <- rbind(ResultsFinal, Resultscad)
#}

write.matrix(ResultsFinal,file=paste0("ResultsRealOTUscad.csv"),sep = ",")