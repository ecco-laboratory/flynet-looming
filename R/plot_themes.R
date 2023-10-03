# setup, aka helper function definition ----

# this is a targets-compatible function script
# so no library()-ing here

## theme objects ----

theme_slides <- function (base_size = 12, base_family = "Graphik", ...) {
  out <- theme_bw(base_size = base_size,
                  base_family = base_family) +
    theme(legend.key = element_blank(),
          legend.background = element_blank(),
          axis.line = element_line(color = "grey40"),
          axis.ticks = element_line(color = "grey40"),
          plot.background = element_rect(fill = "transparent", color = "transparent")) +
    theme(...)
}
