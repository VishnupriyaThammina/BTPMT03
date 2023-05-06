library(sp)  
library(spatstat)  
library(raster)
library(maptools)  # to convert to point pattern

data.shape<-readOGR(dsn="D:\\Downloads\\Downloads\\html_files\\HEATMAPS-20210424T120029Z-001\\2020\\IND_adm1.shp")
plot(data.shape)
new=read.csv("D:\\HONOURS PROJECT\\RESULTS\\GRAPHS\\PREDICTED_24\\ground_24.csv",header = TRUE)
load("D:\\Downloads\\Downloads\\web series\\honors\\Creating-maps-in-R-master\\Creating-maps-in-R-master\\data\\stations.RData")

new<-data.frame(new)
new1<-cbind(as.numeric(new$longitude),as.numeric(new$latitude),as.numeric(new$mae),as.numeric(new$rmse),as.numeric(new$rscore))
colnames(new1)<-c("Lon","Lat","MAE","RMSE","Rscore")

lon=new1[,"Lon"]
lat = new1[,"Lat"]
md = new1[,"MAE"]

plot(lon,lat)

points <- cbind(lon,lat)
library(rgdal)
sputm <- SpatialPoints(points, proj4string=CRS("+proj=utm +zone=10 +datum=WGS84"))
sSp <- as(sputm, "ppp")  # convert points to pp class
Dens <- density(sputm, adjust = 0.2)  # create density object


# coordinates(dat)=~lon+lat
# proj4string(dat)=CRS("+init=epsg:4326")
# dat2=spTransform(dat,CRS(p4))
# bb=bbox(dat2)
# lonx=seq(bb[1,1],  bb[1,2],len=277)
# laty=seq(bb[2,1], bb[2,2],len=349)
# r=raster(list(x=laty,y=lonx,z=md))
# plot(r)
# contour(r,add=TRUE)
# N


