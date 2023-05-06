library(sf)
library(rhdf5)
library(raster)
library(rasterVis)
library(RColorBrewer)
library(openxlsx)
library(raster)
library(ggplot2)
library(rgdal)

shp <- read_sf("C:\\Users\\Santhosh\\Downloads\\newshape\\in\\in.shp")
world_outline <- as(st_geometry(shp), Class="Spatial")
data <- read.csv("D:\\HONOURS\\india_cropped\\hdfview\\modis_3K_india.csv")
data <- data[order(data$lat),]
spg <- data
coordinates(spg) <- ~ lat + long
# coerce to SpatialPixelsDataFrame
gridded(spg) <- TRUE
# coerce to raster
rasterDF <- raster(spg)


dataset <- raster(x=as.numeric(data$lat),y=as.numeric(data$long),z=as.numeric(data$data))
new <-
data1<-rasterToPoints(dataset)
maptheme<-rasterTheme(region = brewer.pal(8,"YlOrRd"))
plt<-levelplot(dataset,margin=F,par.settings=maptheme,main="UI index")
plt+layer(sp.lines(world_outline,col="black",lwd=1.0))




ggplot() +
  geom_polygon(data = shp, aes(x = long, y = lat, group = group),
               fill = "#69b3a2", color = "white") +
  geom_point(data = z, aes(x = long, y = lat),
             size = 0.1) +
  theme_void()
