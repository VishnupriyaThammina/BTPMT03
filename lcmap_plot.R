library(rgdal)
library(raster)
library(rgdal)
library("readxl")

r<-raster("D:\\Downloads\\bda\\Bulk Order landsat8first6months\\testing\\LC08_L2SP_146040_20200109_20200823_02_T1_SR_B1.TIF")
r<-raster('D:\\Downloads\\Downloads\\web series\\honors\\LC08_L1TP_147040_20211001_20211013_02_T1\\LC08_L1TP_147040_20211001_20211013_02_T1_B2.TIF')
r<-raster('D:\\Downloads\\Downloads\\web series\\honors\\LC08_L1TP_147040_20211001_20211013_02_T1\\LC08_L1TP_147040_20211001_20211013_02_T1_B3.TIF')
r<-raster('D:\\Downloads\\Downloads\\web series\\honors\\LC08_L1TP_147040_20211001_20211013_02_T1\\LC08_L1TP_147040_20211001_20211013_02_T1_B4.TIF')

plot(r)
mat<- rasterToPoints(r)
names(r)

gridpts<-read.csv("D:\\HONOURS PROJECT\\DATASET\\Delhi_stn_lulc.csv")
points<- cbind(gridpts$lon,gridpts$lat)
pr1<-projectRaster(r,crs = "+proj=lonlat +lat_1=52 +lon_0=-10 +ellps=WGS84",res=0.001)
colnames(points)<-c("Lan","Lon")
coordinates(points)<-~Lon+Lat
projection(points)<-"+init:epsg=4326"


data.shape<-readOGR(dsn="D:\\HONOURS PROJECT\\DATASET\\shapefile\\Delhi Map-20220401T065437Z-001\\Delhi Map\\Delhi_Boundary.shp")

plot(pr1)
plot(data.shape,add=TRUE)
# points <-SpatialPoints(gridpts[, c('longit', 'latitude')], 
#                         proj4string=CRS('+proj=longlat +datum=WGS84'))
# # and your raster data are not they need to be matched
# pts <- spTransform(points, CRS("+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +datum=WGS84"))

nlayers(r)


out<-extract(pr1, points)
 names(out)<- r






