library(ggplot2)
library(maptools)
library(rgeos)
library(ggmap)
library(scales)
library(RColorBrewer)
library(openxlsx)
library(readxl)

shp <- readShapeSpatial("C:\\Users\\Santhosh\\Downloads\\newshape\\in\\in.shp")
map <- ggplot() + geom_polygon(data = shp, aes(x = long, y = lat, group = group), colour = "black", fill = NA)
map
polygon_dataframe = fortify(shp)
write.csv(polygon_dataframe,"C:\\Users\\Santhosh\\Downloads\\newshape\\in\\x.csv")
