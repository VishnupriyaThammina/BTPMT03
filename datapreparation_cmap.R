library(rgdal)
library(raster)
library(rgdal)
library("readxl")
library("plyr")
library("rgdal")
library(ggmap)
require(ggplot2)
library(pracma)
library(sf)
library(raster)
# 
# original<- getwd()
# setwd("D:/Downloads/bda/Bulk Order landsat8first6months/Landsat 8-9 OLI_TIRS C2 L2")
# zipF <- list.files(path =".", pattern = "*.tar", full.names = TRUE)
# llply(.data = zipF, .fun = untar, exdir = "D:\\Downloads\\bda\\Bulk Order landsat8first6months\\Landsat8_After_extraction")

# setwd(original)
data.shape<-readOGR(dsn="D:\\HONOURS PROJECT\\DATASET\\shapefile\\Delhi Map-20220401T065437Z-001\\Delhi Map\\Delhi_Boundary.shp")
shp <- st_read(dsn="D:\\HONOURS PROJECT\\DATASET\\shapefile\\Delhi Map-20220401T065437Z-001\\Delhi Map\\Delhi_Boundary.shp")
plot(shp)
shp <- st_as_sf(shp)
ggplot() +
  geom_sf(data = shp)

grid <- shp %>% 
  st_make_grid(cellsize = 0.003, what = "centers") %>% # grid of points
  st_intersection(shp)                               # only within the polygon
ggplot() +
  geom_sf(data = shp) +
  geom_sf(data = grid)


grid1 <- data.frame(matrix(unlist(grid), nrow=length(grid), byrow=TRUE))
colnames(grid1)<-c("Lon","Lat")

write.csv(grid1,"D:\\HONOURS PROJECT\\DATASET\\cleanded_landsat8_data\\grid_delhi.csv")

# delhi_stations=read.csv("D:\\HONOURS PROJECT\\DATASET\\lat_lon\\Delhi_stations.csv",header = FALSE)
# colnames(delhi_stations)<-c("id","station_name","PM2.5","Lat","Lon","state")
delhi_stations=read.csv("D:\\HONOURS PROJECT\\DATASET\\lat_lon\\latlong_data.csv")

points<- cbind(delhi_stations$lon,delhi_stations$lat)

original<-getwd()
setwd("D:\\Downloads\\bda\\Bulk Order landsat8first6months\\Landsat8_After_extraction")
temp <- list.files(pattern=".*S._B.")
names<-c()
output<-c()

output1<-c()
# setwd("D:\\Downloads\\bda\\Bulk Order landsat8first6months\\result_testing")

for(i in 1:length(temp)){
  if((i)<length(temp)){
  date<-lubridate::ymd(substring(temp[i],18,25))
  date1<-lubridate::ymd(substring(temp[i+1],18,25))
  if(date==date1){
    r<-raster(temp[i])
    pr1<-projectRaster(r,crs = "+proj=lonlat +lat_1=52 +lon_0=-10 +ellps=WGS84",res=0.001)
    out<-extract(pr1, points)
    names <- c(names,substring(temp[i],45,46))
    output<-cbind(output,out)
  }else{
    r<-raster(temp[i])
    pr1<-projectRaster(r,crs = "+proj=lonlat +lat_1=52 +lon_0=-10 +ellps=WGS84",res=0.001)
    out<-extract(pr1, points)
    names <- c(names,substring(temp[i],45,47))
    output<-cbind(output,out)
    colnames(output)<-names
    dates<-rep(as.character(date),41)
    output<-cbind(output,dates)
    pathrow<-substring(temp[i],11,16)
    pathrows<-rep(pathrow,41)
    output<-cbind(output,pathrows)
    id<-delhi_stations$id
    lat <-delhi_stations$lat
    long <- delhi_stations$lon
    output <- cbind(output,id)
    output <- cbind(output,lat)
    output <- cbind(output,long)
    output1<-rbind(output1,output)
    output<-c()
    names<-c()
    
    # output1<-data.frame(output)
    # filename<-paste(date,".csv",sep="")
    # setwd("D:\\Downloads\\bda\\Bulk Order landsat8first6months\\result_testing")
    # write.csv(output1,filename)
    # setwd("D:/Downloads/bda/Bulk Order landsat8first6months/testing")
  }
  }
  else{
    date<-lubridate::ymd(substring(temp[i],18,25))
    r<-raster(temp[i])
    pr1<-projectRaster(r,crs = "+proj=lonlat +lat_1=52 +lon_0=-10 +ellps=WGS84",res=0.001)
    out<-extract(pr1, points)
    names <- c(names,substring(temp[i],45,47))
    output<-cbind(output,out)
    colnames(output)<-names
    dates<-rep(as.character(date),41)
    output<-cbind(output,dates)
    pathrow<-substring(temp[i],11,16)
    pathrows<-rep(pathrow,41)
    output<-cbind(output,pathrows)
    id<-delhi_stations$id
    lat <-delhi_stations$lat
    long <- delhi_stations$lon
    output <- cbind(output,id)
    output <- cbind(output,lat)
    output <- cbind(output,long)
    output1<-rbind(output1,output)
    output1<-data.frame(output1)
    # filename<-paste(date,".csv",sep="")
    setwd("D:\\Downloads\\bda\\Bulk Order landsat8first6months\\result_testing")
    write.csv(output1,"delhi_lat_lon_final.csv")
    # setwd("D:/Downloads/bda/Bulk Order landsat8first6months/testing")
    
  }
}


