library(sp)
library(rgdal)
library(raster)
library(sf)


delhi_bound<- st_read("D:\\Downloads\\Downloads\\html_files\\HEATMAPS-20210424T120029Z-001\\2020\\IND_adm1.shp")
library(rgdal)
data.shape<-readOGR(dsn="D:\\Downloads\\Downloads\\html_files\\HEATMAPS-20210424T120029Z-001\\2020\\IND_adm1.shp")
plot(data.shape)
proj4string(data.shape)
grid <- makegrid(data.shape, cellsize = 0.365)
grid <- SpatialPoints(grid, proj4string = CRS(proj4string(data.shape)))

plot(data.shape)
plot(grid, pch = ".", add = T)
grid <- grid[data.shape, ]
gd<-data.frame(grid)
write.csv(gd,"D:\\HONOURS PROJECT\\Grid_Delhi_1691.csv")
