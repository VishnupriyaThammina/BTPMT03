library(rgdal)
library(raster)
library(rgdal)
library(ggplot2)
library("readxl")

# setwd("/home/subhojit/Downloads/Landsat8")
r1<-raster("D:\\Downloads\\Downloads\\web series\\honors\\landsat8\\delhi_30min\\LC08_L2SP_147040_20201115_20210315_02_T1\\LC08_L2SP_147040_20201115_20210315_02_T1_SR_B1.TIF")
r2<-raster("D:\\Downloads\\Downloads\\web series\\honors\\landsat8\\delhi_30min\\LC08_L2SP_147040_20201115_20210315_02_T1\\LC08_L2SP_147040_20201115_20210315_02_T1_SR_B2.TIF")
r3<-raster("D:\\Downloads\\Downloads\\web series\\honors\\landsat8\\delhi_30min\\LC08_L2SP_147040_20201115_20210315_02_T1\\LC08_L2SP_147040_20201115_20210315_02_T1_SR_B3.TIF")
r4<-raster("D:\\Downloads\\Downloads\\web series\\honors\\landsat8\\delhi_30min\\LC08_L2SP_147040_20201115_20210315_02_T1\\LC08_L2SP_147040_20201115_20210315_02_T1_SR_B4.TIF")
r5<-raster("D:\\Downloads\\Downloads\\web series\\honors\\landsat8\\delhi_30min\\LC08_L2SP_147040_20201115_20210315_02_T1\\LC08_L2SP_147040_20201115_20210315_02_T1_SR_B5.TIF")
r6<-raster("D:\\Downloads\\Downloads\\web series\\honors\\landsat8\\delhi_30min\\LC08_L2SP_147040_20201115_20210315_02_T1\\LC08_L2SP_147040_20201115_20210315_02_T1_SR_B6.TIF")
r7<-raster("D:\\Downloads\\Downloads\\web series\\honors\\landsat8\\delhi_30min\\LC08_L2SP_147040_20201115_20210315_02_T1\\LC08_L2SP_147040_20201115_20210315_02_T1_SR_B7.TIF")
r10<-raster("D:\\Downloads\\Downloads\\web series\\honors\\landsat8\\delhi_30min\\LC08_L2SP_147040_20201115_20210315_02_T1\\LC08_L2SP_147040_20201115_20210315_02_T1_ST_B10.TIF")

grid<-read.csv("D:\\HONOURS PROJECT\\DATASET\\cleanded_landsat8_data\\grid_delhi.csv")
points<- cbind(grid$Lon,grid$Lat)
pr1<-projectRaster(r1,crs = "+proj=lonlat +lat_1=52 +lon_0=-10 +ellps=WGS84",res=0.001)
out<-extract(pr1, points)
output<-cbind(points,out)
colnames(output)<-c("Lon","Lat","B1")
map<-ggplot(output,aes(Lon,Lat,z=B1))+geom_contour_filled()+ylab("Latitude")+xlab("Longitude")




data.shape<-readOGR(dsn="D:\\HONOURS PROJECT\\DATASET\\shapefile\\Delhi Map-20220401T065437Z-001\\Delhi Map\\Delhi_Boundary.shp")

crs(data.shape)
z<-"+proj=latlong +datum=WGS84 +no_defs"
r1<- projectRaster(r1, crs = z)

plot(data.shape,col="black",add=TRUE)
plot(r1)

map<-ggplot()+geom_polygon(data=data.shape,aes(x=long,y=lat,group=group),color="black",fill=NA)

rr1plot(data.shape, add=r1)

s <- stack(r1,r2,r3,r4) 
plotRGB(s, scale=65535)
plot(s)
boundary = raster(ymx=28.91, xmn=76.78, ymn=28.36, xmx=77.39)
plotRGB(s - 5000, scale=12000, zlim=c(0, 12000))

boundary = projectExtent(raster(data.shape), s@crs)

champaign = crop(s, boundary)
plot(champaign)
plotRGB(champaign)

masked <- mask(x = s, mask = data.shape)
plot(masked)


plot(s)
plot(r3)
data.shape<-readOGR(dsn="/home/subhojit/Downloads/Mainak Sir/Delhi pollution/Map/Delhi_Boundary-SHP/Delhi_Boundary.shp")
proj4string(r1)<-proj4string(data.shape)
masked <- mask(x = r1, mask = data.shape)
plot(masked)
plot(data.shape)
proj4string(data.shape)
grid <- makegrid(data.shape, cellsize = 0.1)
grid <- SpatialPoints(grid, proj4string = CRS(proj4string(data.shape)))
plot(grid, pch = ".", add = T)

plot(s)
mat<- rasterToPoints(r)
names(r)

gridpts<-read.csv('/home/subhojit/Downloads/Data_Analysis_Prisma/Dataset_V2/Newgridpoints.csv')
points<- cbind(gridpts$Lat,gridpts$Lon)
nlayers(r)
for(i in 1:nlayers(r)){
  band<-r[[i]]
  #save raster in a separate file
  writeRaster(band,paste('band',i,'.tif', sep=''))
}