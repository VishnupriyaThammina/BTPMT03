library("dplyr")
library("raster")

harvesine<-function(lon1, lat1, lon2, lat2){
  rad <-pi / 180
  R <- 6378.1
  dlon <- (lon2- lon1) * rad
  dlat <- (lat2-lat1) * rad
  a <- (sin(dlat / 2)) ** 2 + cos(lat1 * rad) * cos(lat2 * rad) * ((sin(dlon / 2)) ** 2)
  c = 2 * atan2(sqrt(a), sqrt(1 - a))
  d = R * c
  return(d)
}

idwr<-function(x,y,z,xi,yi){
  lstxyzi<- c()
  print(length(xi))
  for (p in 1:length(xi)){
    print(p)
    lstdist <- c()
    for (s in 1:length(x)){
      d = (harvesine(y[s], x[s], yi[p], xi[p]))
      lstdist<-c(lstdist,d)
      print(d)
    }
    sumsup = 1 / (lstdist)** 2
    suminf = sum(sumsup)
    sumsup = sum((sumsup)*(z))
    u = sumsup / suminf
    # xyzi = cbind(xi[p], yi[p], u)
    # lstxyzi<-rbind(lstxyzi,xyzi)
    lstxyzi<-rbind(lstxyzi,u)
  }
  return(lstxyzi)
}

final <- function(r){
  u <- raster(r)
  temp <- as.data.frame(u,xy=T)
  colnames(temp)<-c("x","y","z")
  indx <- which(is.na(temp[3]))
  notindx <- which(!is.na(temp[3]))
  ts_coord_x<-as.numeric(as.character(temp$y[indx]))
  ts_coord_y<-as.numeric(as.character(temp$x[indx]))
  tr_coord_z<-as.numeric(as.character(temp$z[notindx]))
  tr_coord_x<-as.numeric(as.character(temp$y[notindx]))
  tr_coord_y<-as.numeric(as.character(temp$x[notindx]))
  st<-data.frame(idwr(tr_coord_x,tr_coord_y,tr_coord_z,ts_coord_x,ts_coord_y))
  temp$z[indx]<-st
  return(temp)
}


o<-final("C:\\Users\\Santhosh\\Desktop\\india_tiff\\2019-02-07.tif")
r<-raster("C:\\Users\\Santhosh\\Desktop\\result_aod\\2019-02-07.tif")
u <- raster(r)
temp <- as.data.frame(u,xy=T)
colnames(temp)<-c("x","y","z")
k<-temp[3]
factorInd<-sapply(k,is.factor)
