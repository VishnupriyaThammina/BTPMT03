library(ggplot2)
library(maptools)
library(rgeos)
library(ggmap)
library(scales)
library(RColorBrewer)
library(openxlsx)
library(readxl)
set.seed(8000)
shp <- readShapeSpatial("D:\\Downloads\\Downloads\\html_files\\HEATMAPS-20210424T120029Z-001\\2020\\IND_adm1.shp")
z <- read_excel("D:\\HONOURS PROJECT\\DATASETS_FOR_CODE\\YEARLY AVERAGE FOR MAP ALL INDIA\\testing_2020.xlsx")
# z1 <- read_excel("D:\\Downloads\\Downloads\\html_files\\HEATMAPS-20210424T120029Z-001\\2020\\drive-download-20210510T113816Z-001\\dbscan_2020.xlsx")
z1 <- read_excel("D:\\Downloads\\Downloads\\html_files\\HEATMAPS-20210424T120029Z-001\\2020\\c.xlsx")
score = z$pm25
state_data = data.frame(id=shp$ID_1, NAME_1=shp$NAME_1,score)
fortify_shape = fortify(shp, region = "ID_1")

Merged_data = merge(fortify_shape, state_data, by="id", all.x=TRUE)
Map_plot = Merged_data[order(Merged_data$order), ]
clusters = z1$label

ggplot() +
  geom_polygon(data = Map_plot,
               aes(x = long, y = lat, group = group),
               color = "black", size = 0.5) +
  scale_fill_gradientn(colors = rev(rainbow(4)),
                       breaks = c(30,60,90,120),
                       trans = "log10"
  )+
  labs(fill = "PM2.5")+
  theme(legend.background = element_rect(fill="lightblue", 
                                         size=0.1, linetype="solid",color = "darkblue"),legend.position = c(0.8,0.25),legend.title = element_text(size = rel(0.75)))+
  geom_point(data = z1, aes(x = longitude, y = latitude,colour = factor(label)), size = 2)+
  geom_bin2d(bins=10)+
  labs(color="Hierarchical")+
  coord_map()

