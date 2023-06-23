# This script plots the simulation results
library(tidyverse)

# Set plotting theme
theme_set(theme(panel.grid = element_blank()))

# Read in the data
sim_results <- read_csv("sim_results/best_results.csv")

# Plot the results for p = 500
sim_results %>%
    group_by(setting, data_id) %>%
    arrange(bic, .by_group = TRUE) %>%
    filter(row_number() == 1) %>%
    filter(str_detect(setting, "dim500")) %>%
    mutate(cor = str_extract(setting, "rho([0-9]+\\.[0-9]+)", group = 1),
        penalty = str_extract(setting, "covid_(.*)-", group = 1)) %>%
    ggplot(aes(x = n_nz - tp, y = cor, col = penalty)) +
    geom_boxplot(position = position_dodge(width = 0.9)) +
    labs(x = "Number of false positives", y = "Correlation among predictors", title = "Results for p = 500") 
    
ggsave("sim_results/p500.png", width = 20, height = 20, units = "cm") 

# For each unique combination of setting and data_id, select the model with the lowest BIC
sim_results %>%
    group_by(setting, data_id) %>%
    arrange(bic, .by_group = TRUE) %>%
    filter(row_number() == 1) %>%
    mutate(correlation = str_extract(setting, "rho([0-9]+\\.[0-9]+)", group = 1),
        penalty = str_extract(setting, "cov.*_(.*)-", group = 1),
        `# random effects` = str_extract(setting, "random([0-9])", group = 1),
        cov_str = str_extract(setting, "cov(.*)_", group = 1)) %>%
    ggplot(aes(x = n_nz - tp, y = cov_str, col = penalty)) +
    geom_boxplot(position = position_dodge(width = 0.9)) +
    labs(x = "Number of false positives", y = "Random effects' covariance structure", title = "Results for p = 1000") +
    facet_grid(`# random effects`~correlation, labeller = label_both, scales = "free_y") +
    theme_bw() +
    scale_color_discrete(breaks=c('scad', 'lasso'))

ggsave("sim_results/p1000.png", width = 30, height = 30, units = "cm") 



