library(SpiecEasi)
library(SPRING)

data(amgut1.filt)
p = ncol(amgut1.filt) # the number of nodes.
e = 2*p # the number of edges is set as twice the number of nodes.

gtypes <- c("band", "cluster", "scale_free", "erdos_renyi", "hub", "block")
simulated_data <- vector(mode='list', length=length(gtypes))
nd = 100 #Number of data sets to generate under each setting

i = 1
set.seed(12345) # set the seed number for make_graph part.
for (gtype in gtypes) {
  
  datasets = list(mode = 'list', length = nd)
  for (j in 1:nd) {
    
    # Create a taxa-taxa correlation matrix
    graph_amgut <- SpiecEasi::make_graph(gtype, p, e) # adjacency matrix. 1: edge, 0: no edge.
    Prec_amgut  <- SpiecEasi::graph2prec(graph_amgut) # precision matrix. inverse of covariance.
    Cor_amgut   <- cov2cor(SpiecEasi::prec2cov(Prec_amgut)) # correlation matrix.
    
    # Generate synthetic data based on real amgut data and desired correlation struct
    X_amgut <- synthData_from_ecdf(amgut1.filt, Sigma = Cor_amgut, n = 120)
    
    # Imput pseudo counts to 0's and take log ratio, using last column as reference
    min_nz = min(X_amgut[X_amgut != 0])
    X_amgut[X_amgut == 0] <- min_nz/2
    X_amgut <- log(X_amgut)
    X_amgut <- X_amgut - X_amgut[,ncol(X_amgut)]

    datasets[[j]] = X_amgut
    write.csv(X_amgut, paste0("data/OTU/spring/data", gtype, "_", j, ".csv"), row.names = FALSE)
    print(j)
  }
  simulated_data[[i]] = datasets
  i <- i +1 
  
}






















