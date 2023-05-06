library("readxl")
library("openxlsx")
library("dplyr")
require (reshape2)
require (ggplot2)


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
# IDW function with power 2 (x=lat, y=lon)
idwr<-function(x,y,z,xi,yi){
  lstxyzi<- c()
  for (p in 1:length(xi)){
    lstdist <- c()
    for (s in 1:length(x)){
      d = (harvesine(y[s], x[s], yi[p], xi[p]))
      lstdist<-c(lstdist,d)
    }
    sumsup = 1 / (lstdist)** 2
    suminf = sum(sumsup)
    sumsup = sum((sumsup)*(z))
    u = sumsup / suminf
    print(u)
    xyzi = cbind(xi[p], yi[p], u)
    lstxyzi<-rbind(lstxyzi,xyzi)
  }
  return(lstxyzi)
}

stn_loc <-read.csv("D:\\HONOURS PROJECT\\DATASET\\lat_lon\\Delhi_stations.csv",header=FALSE)
colnames(stn_loc)<-c("id","station","pm25","latitude","longitude","state")
delhi_nas <- read.csv("D:\\HONOURS PROJECT\\DATASET\\lat_lon\\delhi_pm25.csv")
for(i in 1:22){
  indx <- apply(delhi_nas[-c(1)][i,], 2, function(x) any(is.na(x) | is.infinite(x)))
  if( 'TRUE' %in% indx){
    ts_coord_x<-as.numeric(as.character(stn_loc$latitude[indx]))
    ts_coord_y<-as.numeric(as.character(stn_loc$longitude[indx]))
    tr_coord_z<-delhi_nas[-c(1)][i,][!indx]
    colnames(tr_coord_z)<-NULL
    tr_coord_z <- as.numeric(tr_coord_z)
    tr_coord_x<-as.numeric(as.character(stn_loc$latitude[!indx]))
    tr_coord_y<-as.numeric(as.character(stn_loc$longitude[!indx]))
    st<-data.frame(idwr(tr_coord_x,tr_coord_y,tr_coord_z,ts_coord_x,ts_coord_y))
    dd<-t(delhi_nas[-c(1)][i,])
    dd[which(indx=="TRUE")]<-st
    delhi_nas[-c(1)][i,]<-dd
  }
  
  }
write.csv(delhi_nas,"D:\\HONOURS PROJECT\\DATASET\\lat_lon\\delhi_pm25_cleaned.csv")

data<-read.csv("D:\\HONOURS PROJECT\\DATASET\\cleanded_landsat8_data\\delhi_landsat8_station_data_cleaned_final.csv")
b1<-data[,c(2)]
b2<-data[,c(3)]
b3<-data[,c(4)]
b4<-data[,c(5)]
# hist(data1, col = "steelblue", frame = FALSE)
plot(density(b1))                  # Plot density of x
lines(density(b2), col = "red")                      # Overlay density of y
lines(density(b3), col = "green")  
lines(density(b4), col = "brown")    

# data1$new<-c("B1","B2")
# data2<-melt(data1)
# ggplot(data2, aes (value)) +
#   geom_density(color = variable)

