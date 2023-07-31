# Rosa's results --------------
r_lasso1 <- read_csv("sim_results/rosa/ResultsFinalLasso11.csv") %>%
  rename(setting = `...1`)
r_lasso2 <- read_csv("sim_results/rosa/ResultsFinalLasso22.csv") %>%
  rename(setting = `...1`)
r_scad1 <- read_csv("sim_results/rosa/ResultsFinalSCAD1.csv") %>%
  rename(setting = `...1`)
r_scad2 <- read_csv("sim_results/rosa/ResultsFinalSCAD2.csv") %>%
  rename(setting = `...1`)

r_lasso <- bind_rows(r_lasso1, r_lasso2) %>% 
  mutate(penalty = "lasso") %>%
  select(!starts_with("random"))

r_scad <- bind_rows(r_scad1, r_scad2) %>%
  mutate(penalty = "scad") %>%
  select(!starts_with("random"))

r_all <- bind_rows(r_lasso, r_scad)

# Remove massive data frames to save space
rm(list = c("r_lasso1", "r_lasso2", "r_scad1", "r_scad2", "r_lasso", "r_scad") ) 



## nz = 5 ===========

### false positives #################

r_all %>%
  filter(str_detect(setting, "nz5")) %>%
  mutate(
    cor = str_extract(setting, "rho([0-9]+\\.[0-9]+)", group = 1),
    p = factor(
      str_extract(setting, "dim([0-9]+)", group = 1),
      levels = c('500', "1000")
    )
  ) %>%
  ggplot(aes(y = (n_nonzero_B-TP)/(as.numeric(as.character(p))-5), col = penalty)) +
  geom_boxplot(position = position_dodge(width = 0.9)) +
  facet_grid(cor ~ p, labeller = "label_both") +
  labs(y = "False positive rate") +
  scale_y_continuous(labels = scales::percent) +
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank())


### true positives #################

r_all %>%
  filter(str_detect(setting, "nz5")) %>%
  mutate(
    cor = str_extract(setting, "rho([0-9]+\\.[0-9]+)", group = 1),
    p = factor(
      str_extract(setting, "dim([0-9]+)", group = 1),
      levels = c('500', "1000")
    )
  ) %>%
  ggplot(aes(y = TP, col = penalty)) +
  geom_boxplot(position = position_dodge(width = 0.9)) +
  facet_grid(cor ~ p, labeller = "label_both") +
  labs(y = "True positives (out of 5)") +
  scale_y_continuous(labels = scales::label_number(accuracy=1), limits = c(0,5)) +
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank())




## p=1000 and nz = 10 ====================

### false positives ######################

r_all %>%
  filter(str_detect(setting, "nz10")) %>%
  filter(!(str_detect(setting, "sym") & penalty == "scad")) %>%
  mutate(
    correlation = str_extract(setting, "rho([0-9]+\\.[0-9]+)", group = 1),
    `# random effects` = str_extract(setting, "random([0-9])", group = 1),
    cov_str = factor(
      str_extract(setting, "cov(.*)", group = 1),
      levels = c("id", "diag", "sym")
    )
  ) %>%
  ggplot(aes(y = (n_nonzero_B-TP)/990, x = cov_str, col = penalty)) +
  geom_boxplot(position = position_dodge(width = 0.9), width = .3) +
  labs(y = "False positive rate", x = "Random effect covariance") +
  facet_grid(
    correlation ~ `# random effects`,
    labeller = label_both,
    space = "free", 
    scales = "free_x"
  ) + #Get rid of space argument if you want box plot widts to automatically adjust to space in facet
  scale_y_continuous(labels = scales::percent, limits = c(0,.05)) #Not observing a couple of outliers 

# Change label of "correlation" to "correlation between predictors" or change to "\rho" 
# and change "# random effects to q"
  
  
      
### true positives #########################

r_all %>%
  filter(str_detect(setting, "nz10")) %>%
  filter(!(str_detect(setting, "sym") & penalty == "scad")) %>%
  mutate(
    correlation = str_extract(setting, "rho([0-9]+\\.[0-9]+)", group = 1),
    `# random effects` = str_extract(setting, "random([0-9])", group = 1),
    cov_str = factor(
      str_extract(setting, "cov(.*)", group = 1),
      levels = c("id", "diag", "sym")
    )
  ) %>%
  ggplot(aes(y = TP, x = cov_str, col = penalty)) +
  geom_boxplot(position = position_dodge(width = 0.9), width = .3) +
  labs(y = "True positives (out of 10)", x = "Random effect covariance") +
  facet_grid(
    correlation ~ `# random effects`,
    labeller = label_both,
    scales = "free_x",
    space = "free"
  ) + #Get rid of space argument if you want box plot width to automatically adjust to space in facet
  scale_y_continuous(labels = scales::label_number(accuracy=1))  

# Change label of "correlation" to "correlation between predictors" or change to "\rho" 
# and change "# random effects to q"

















