## helper functions ----

r2 <- function (y, x) sum(y * x)^2 / (sum(y^2) * sum(x^2))

## calculate partial r-squared from dataframe ----

# Note now that this takes STRINGS so that covar_cols can take formula syntax
calc_partial_r2 <- function(in_data, y_col, x_col, covar_cols) {
  stopifnot(is.character(y_col), is.character(x_col), is.character(covar_cols))
  if (length(covar_cols) > 1) {
    covar_cols <- paste(covar_cols, collapse = "+")
  }
  
  out <- in_data %>% 
    mutate(resid_y_covar = lm(as.formula(paste(y_col, covar_cols, sep = "~"))) %>% pluck("residuals"),
           resid_x_covar = lm(as.formula(paste(x_col, covar_cols, sep = "~"))) %>% pluck("residuals")) %>% 
    summarize(partial_r2 = r2(resid_y_covar, resid_x_covar)) %>% 
    pull(partial_r2)
  
  return (out)
}

calc_perm_partial_r2 <- function (in_perm, x_col, covar_cols, resid_y_col, resid_x_col, fitted_y_col) {
  # coming in, currently expect resid_y_col to be permuted
  stopifnot(is.character(x_col), is.character(covar_cols), is.character(resid_y_col), is.character(resid_x_col), is.character(fitted_y_col))
  if (length(covar_cols) > 1) {
    covar_cols <- paste(covar_cols, collapse = "+")
  }
  
  # Uses Freeman & Lane (1983) but as described by Anderson & Robinson (2001)
  # assumes the residuals have _already_ been calculated on the real data
  # as these don't need to be calculated on each permutation
  
  # To generalize this to multiple covariate columns
  # My tiny brain can no longer use the derived formula for a single OLS coefficient
  # Get residualized permuted y from lm(y_perm ~ covar_cols)
  
  out <- in_perm %>% 
    # add the permuted y~covar residuals onto the original fitted y~covar values
    mutate(y_perm = !!sym(fitted_y_col) + !!sym(resid_y_col),
           # get the residuals of y (permuted) ~ covars (real)
           resid_y_covar_perm = lm(as.formula(paste("y_perm", covar_cols, sep = "~")), data = .) %>% pluck("residuals")) %>% 
    summarize(r2_partial_perm = r2(resid_y_covar_perm, !!sym(resid_x_col))) %>%
    pull(r2_partial_perm)
  
  return (out)
}

## permutation test wrapper ----

# IT TAKES THEM AS STRINGS NOW!!!
perm_partial_r2 <- function (in_data, y_col, x1_col, x2_col, covar_cols = NULL, times) {
  stopifnot(is.character(y_col), is.character(x1_col), is.character(x2_col))
  
  # First: Calculate the relevant values from the _real_ data
  # so that this only gets done once, not per permutation
  in_data %<>% 
    # Ha! this works
    mutate(resid_y_x2 = lm(as.formula(paste(y_col, paste(c(x2_col, covar_cols), collapse = "+"), sep = "~"))) %>% pluck("residuals"),
           fitted_y_x2 = lm(as.formula(paste(y_col, paste(c(x2_col, covar_cols), collapse = "+"), sep = "~"))) %>% pluck("fitted.values"),
           resid_x1_x2 = lm(as.formula(paste(x1_col, paste(c(x2_col, covar_cols), collapse = "+"), sep = "~"))) %>% pluck("residuals"),
           resid_y_x1 = lm(as.formula(paste(y_col, paste(c(x1_col, covar_cols), collapse = "+"), sep = "~"))) %>% pluck("residuals"),
           fitted_y_x1 = lm(as.formula(paste(y_col, paste(c(x1_col, covar_cols), collapse = "+"), sep = "~"))) %>% pluck("fitted.values"),
           resid_x2_x1 = lm(as.formula(paste(x2_col, paste(c(x1_col, covar_cols), collapse = "+"), sep = "~"))) %>% pluck("residuals"))
  
  # Next: permute the relevant outcome-predictor residuals
  # DO NOT need to permute the actual outcome variable
  # because the permuted outcome variable will be constructed from the real covariates and the permuted residuals
  # per Freedman & Lane's (1983) formula via Anderson & Robinson (2001)
  out <- in_data %>% 
    permutations(permute = c(starts_with("resid_y")), times = times) %>% 
    # needs to be named using tidymodels tune convention to use tune metrics collecting later
    mutate(partial.r2_x1 = map_dbl(splits,
                                   \(x) x %>% 
                                     analysis() %>% 
                                     calc_perm_partial_r2(x_col = x1_col,
                                                          covar_cols = c(x2_col, covar_cols), 
                                                          resid_y_col = "resid_y_x2",
                                                          resid_x_col = "resid_x1_x2",
                                                          fitted_y_col = "fitted_y_x2"),
                                   .progress = "permuting confusion regression partial r-squared for x1"),
           partial.r2_x2 = map_dbl(splits,
                                   \(x) x %>% 
                                     analysis() %>% 
                                     calc_perm_partial_r2(x_col = x2_col,
                                                          covar_cols = c(x1_col, covar_cols), 
                                                          resid_y_col = "resid_y_x1",
                                                          resid_x_col = "resid_x2_x1",
                                                          fitted_y_col = "fitted_y_x1"),
                                   .progress = "permuting confusion regression partial r-squared for x2"))
  
  return (out)
}
