library(ggplot2)



data <- read.csv("combined.csv")
delhiLat=77.1112615
delhiLon=28.7500499



# convert timestamp column to datetime object
data$time <- as.POSIXct(data$time)

# create plot
plot1 <- ggplot(data, aes(x = time, y = prectot.60,color="Delhi",legend)) +
  geom_line() + 
  xlab("Date and Time") + 
  ylab("Measured Values") + 
  ggtitle("Monitoring Station in Delhi")

plot2 <- ggplot(data, aes(x = time, y = prectot.240,color="Kannur",legend)) +
  geom_line() + 
  xlab("Date and Time") + 
  ylab("Measured Values") + 
  ggtitle("Monitoring Station in Kannur")
combined_plot <- plot1 + plot2 +
  facet_wrap(~ggtitle, ncol = 2)

ggplot() +
  geom_line(data = data, aes(x = time, y = prectot.60, color = "Delhi")) +
  geom_line(data = data, aes(x = time, y = prectot.80, color = "Hebbal")) +
  
  # Add x-axis and y-axis labels.
  xlab("Time") +
  ylab("Value") +
  
  # Add a title.
  ggtitle("Time Series Plot") +
  
  # Add a subtitle (optional).
  labs(subtitle = "Subtitle") +
  
  # Create a legend for the color scale.
  scale_color_manual(name = "City",
                     values = c(Delhi = "blue", Hebbal = "red"),
                     labels = c("Delhi", "Hebbal Bengaluru"))
  ggsave("delhi_hebbal.png",width = 8, height = 6, dpi = 300)

dev.off()
