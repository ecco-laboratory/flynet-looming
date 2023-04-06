# This is a targets-compatible function definition script
# Which means it should only be called under the hood by tar_make()
# and all the packages are loaded ELSEWHERE! Not in the body of this script.

## functions ----

# in general, when these plots output ggplots,
# they will output what I consider the "bare minimum" level of stylistic classing up
# only the things that would NEVER get overwritten based on different thematic vibes

plot_confusion_matrix <- function (confusions, row_col, col_col, fill_col, level_order = NULL) {
  row_col <- enquo(row_col)
  col_col <- enquo(col_col)
  fill_col <- enquo(fill_col)
  
  if (!is.null(level_order)) {
    confusions %<>%
      mutate(across(c(!!row_col, !!col_col), \(x) factor(x, levels = level_order)))
  }
  
  out <- confusions %>% 
    ggplot(aes(x = col_col, y = fct_rev(!!row_col))) + 
    geom_raster(aes(fill = !!fill_col)) + 
    guides(x = guide_axis(angle = 45))
  
  return (out)
}
