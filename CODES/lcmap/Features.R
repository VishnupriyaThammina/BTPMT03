library(rgdal)
library(raster)
library(rgdal)
library("readxl")
setwd("/home/subhojit/Downloads/Data_Analysis_Prisma/ExtractRcsv")
r<-raster('/home/subhojit/Downloads/Data_Analysis_Prisma/PRS_L2C_STD_20211004053414_20211004053418_0001_HCO_FULL.tif')
plot(r)
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

out1<-c()
for(i in 2:37){
  band<- r[[i]]
  out<-extract(band, points)
  names(out)<- band
  out1<-cbind(out1,out)
  #out1<-data.frame(test.dat$latitude, test.dat$longitude, out)
  #form = sprintf('band_%s.csv', i)
  #write.csv(out1,file=form)
}

groundpm<-read.csv('/home/subhojit/Downloads/Data_Analysis_Prisma/Dataset_V2/SGS/Newdataset.csv')
pm<-groundpm$Value
out_arr<-data.frame(which(is.na(out1[,1])))
pm1<-pm[!y]
out1<-na.omit(out1)
y <- complete.cases(out1)

write.csv(out1, 'Feature_Val.csv')


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
    xyzi = cbind(xi[p], yi[p], u)
    lstxyzi<-rbind(lstxyzi,xyzi)
  }
  return(lstxyzi)
}
out1<-data.frame(out1)
count = 0
for (i in 1:4900){
  #print(length(out1[i,]))
  indx <- apply(out1[i,], 2, function(x) any(is.na(x) | is.infinite(x)))
  s<-length(indx[indx== TRUE])
  #print(s)
  if (s<30){
    print(s)
    #count= count+1
    #if( 'TRUE' %in% indx){
      #ts_x<-as.numeric(gridpts$Lat[!indx])
      #ts_y<-as.numeric(gridpts$Lon[!indx])
      #tr_z<-as.numeric(gridpts$Lat[indx]
      #tr_x<-
      #tr_y<-
    }
  }
}
###### 
dim(comp_data)[1]
count = 0
for (i in 1:4900){
  #print(length(out1[i,]))
  indx <- apply(comp_data[i,], 2, function(x) any(is.na(x) | is.infinite(x)))
  #print(out1[i])
  #s<-length(indx[indx== TRUE])
  #print(s)
  if (s==36){
    comp_data<-comp_data[-c(i),]
  }
  else if (s<30) {
    #print(s)
    count=count+1
  }
}
comp_data<-comp_data[-c(1),]
######
comp_data<-cbind(out1,pm,gridpts)
for (i in 1:36){
  #print(length(out1[i,]))
  indx <- apply(out1[i], 2, function(x) any(is.na(x) | is.infinite(x)))
  s<-length(indx)
  print(s)
  }
  s<-length(indx[indx== TRUE])


comp_data<-na.omit(comp_data)


write.csv(comp_data,'complete_dataset2.csv')






