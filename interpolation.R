library("rgdal")
library(ggmap)
require(ggplot2)
library(pracma)
library(sf)
library(raster)
data.shape<-readOGR(dsn="D:\\Downloads\\Downloads\\web series\\honors\\India_State_Shapefile\\India_State_Shapefile\\India_State_Boundary.shp")


shp <- st_read(dsn="D:\\Downloads\\Downloads\\web series\\honors\\India_State_Shapefile\\India_State_Shapefile\\India_State_Boundary.shp")
plot(shp)
shp <- st_as_sf(shp)
ggplot() +
geom_sf(data = shp)

grid <- shp %>% 
  st_make_grid(cellsize = 0.1, what = "centers") %>% # grid of points
  st_intersection(shp)                               # only within the polygon
ggplot() +
geom_sf(data = shp) +
geom_sf(data = grid)


grid1 <- data.frame(matrix(unlist(grid), nrow=length(grid), byrow=TRUE))
colnames(grid1)<-c("Lon","Lat")


# ggplot()+
#   geom_point(data = grid1, aes(x = grid1$X1, y = grid1$X2),shape = 4)+
#   labs(x ="x", y = "y")+
#   scale_color_gradientn(colors = terrain.colors(10))+
#   theme_bw()
# =======================================================================
harvesine<-function(lon1, lat1, lon2, lat2){
  rad <-pi / 180
  R <- 6378.1
  dlon <- abs(lon2- lon1) * rad
  dlat <- abs(lat2-lat1) * rad
  a <- (sin(dlat / 2)) * 2 + cos(lat1 * rad) * cos(lat2 * rad) * ((sin(dlon / 2)) * 2)
  c = 2 * atan2(sqrt(a), sqrt(1 - a))
  d = R * c
  return(d)
}
# IDW function with power 2 
idwr<-function(x,y,z,xi,yi){
  lstxyzi<- c()
  for (p in 1:length(xi)){
    lstdist <- c()
    for (s in 1:length(x)){
      d = (harvesine(y[s], x[s], yi[p], xi[p]))
      # print(d)
      lstdist<-c(lstdist,d)
    }
    sumsup = 1 / (lstdist)** 2
    suminf = sum(sumsup)
    sumsup = sum((sumsup)*(z))
    u = sumsup / suminf
    xyzi = cbind(xi[p], yi[p], u)
    lstxyzi<-rbind(lstxyzi,xyzi)
  }
  return(lstxyzi)
}
# ======================================================================
new=read.csv("D:\\HONOURS PROJECT\\RESULTS\\GRAPHS\\PREDICTED_24\\spectral_24.csv",header = TRUE)
new<-data.frame(new)
new1<-cbind(as.numeric(new$longitude),as.numeric(new$latitude),as.numeric(new$rscore),as.numeric(new$mae),as.numeric(new$rmse))
colnames(new1)<-c("Lon","Lat","Rscore","MAE","RMSE")
new1<-data.frame(new1)

new2<-cbind(new1$Lon,new1$Lat,new1$Rscore)
colnames(new2)<-c("Lon","Lat","Rscore")
new2<-data.frame(new2)


  x<- new2$Lat
  y<-new2$Lon
  z<-new2$Rscore
  xi<-grid1$Lat
  yi<-grid1$Lon
  st<-data.frame(idwr(x,y,z,xi,yi))
 colnames(st)<-c("Lat","Lan","Level")
  ggplot(st, aes(Lan,Lat)) + 
    coord_equal() + 
    xlab('Longitude') + 
    ylab('Latitude') + 
    stat_density2d(aes(fill ="Level"), alpha = .5,
                   geom = "polygon", data = st) + 
    scale_fill_viridis_c() + 
    theme(legend.position = 'none')
map<-ggplot(st,aes(V2,V1))+geom_raster(aes(fill=u))+geom_polygon(data=data.shape,aes(x=long,y=lat,group=group),color="black",fill=NA)+scale_fill_gradientn(name = "RSCORE",colours=terrain.colors(10))

map<-ggplot(st,aes(V2,V1))+stat_density2d(aes(fill=..u..),data=st)+geom_polygon(data=data.shape,aes(x=long,y=lat,group=group),color="black",fill=NA)+scale_fill_gradientn(name = "RSCORE",colours=terrain.colors(10))

# grid1<-data.frame(unlist(apply(grid, 2, list), use.names = FALSE))

# cs <- c(3.28084, 3.28084)*6000
# grdpts <- makegrid(shp, cellsize = cs)
# spgrd <- SpatialPoints(grdpts, proj4string = CRS(proj4string(shp)))
# spgrdWithin <- SpatialPixels(spgrd[shp,])
# plot(spgrdWithin, add = T)
