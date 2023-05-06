library(ggplot2)
library(maptools)
library(rgeos)
library(ggmap)
library(scales)
library(RColorBrewer)
library(openxlsx)
library(readxl)
set.seed(8000)
c
z1 <- read.csv("D:\\HONOURS PROJECT\\DATASET\\NearestNbrsLatLonAllStns.csv")
colnames(z1)<-c("lat","lon")


ggplot() +
  geom_polygon(data = data.shape,
               aes(x = long, y = lat, group = group),
               fill= "orange2",color="black", size = 0.5) +
  geom_point(data = z1, aes(x = lon, y = lat), size = 1.5,color="blue")+
  coord_map()+
  theme_void()

