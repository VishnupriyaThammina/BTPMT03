library(ggplot2)
library(RColorBrewer)
library(ggmap)
library(maps)
library(rgdal)
library(scales)
library(maptools)
library(gridExtra)
library(rgeos)

states_shape = readShapeSpatial("D:\\Downloads\\Downloads\\html_files\\HEATMAPS-20210424T120029Z-001\\2020\\IND_adm1.shp")

 print(states_shape$NAME_1)

# new=read.csv("D:\\Downloads\\Downloads\\web series\\honors\\wanted.csv",header = TRUE)
new=read.csv("D:\\Downloads\\Downloads\\web series\\honors\\wanted_std.csv",header = TRUE)

PM2.5 = new$pm25
State_data = data.frame(id=states_shape$ID_1, NAME_1=states_shape$NAME_1,PM2.5)
fortify_shape = fortify(states_shape, region = "ID_1")
Merged_data = merge(fortify_shape, State_data, by="id", all.x=TRUE)
Map_plot = Merged_data[order(Merged_data$order), ]
ggplot() +
  geom_polygon(data = Map_plot,
               aes(x = long, y = lat, group = group,fill = PM2.5),
               color = "black", size = 0.5) +
  coord_map()+
  scale_fill_gradient(name="Standard Deviation", limits=c(0,100), low = 'yellow', high = 'red')+
  xlab('Longitude')+
  ylab('Latitude')

