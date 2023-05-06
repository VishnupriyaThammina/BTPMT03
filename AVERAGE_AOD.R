library(ncdf4)
library(chron)
library(readxl)
library(stringr)
library(dplyr)
library(raster)

setwd("C:\\Users\\Santhosh\\Desktop\\modis_geotif")
f <- list.files(getwd())
# f <- f[order(as.Date(substring(f,1,7),"%y%d"))]
f<-f[order(substring(f,1,7))]
f
# rootname <- substring(f,7,15)
rootname <- substring(f,1,7)
dates<-rootname
dates<-as.Date(rootname,"%d%b%Y")
s = list()
k = 1
for(i in 1:(length(dates)-1)){
  print(i)
  if(dates[i]==dates[i+1]){
    s[k]=f[i]
    k = k+1
  }
  else{
    s[k]=f[i]
    # assign(paste(dates[i],i,sep=""),stack(s))
    resultingStack <- stack()
    for(j in 1:length(s)){
      print(j)
      tempraster <- crop(raster(s[[j]]),a)
      resultingStack <- stack(resultingStack,tempraster)
    }
    # STACK1 <- stack(s)
    if(length(s)>1){
    # mean2 <- calc(STACK1, fun = mean, na.rm = T)
      mean2 <- calc(resultingStack, fun = mean, na.rm = T)
    # setwd("C:\\Users\\Santhosh\\Desktop\\result_aod")
    setwd("C:\\Users\\Santhosh\\Desktop\\modis_averae")
    writeRaster(x = mean2, filename =paste(dates[i],".tif",sep=""), driver = "GeoTiff")
    # setwd("C:\\Users\\Santhosh\\Desktop\\test")
    setwd("C:\\Users\\Santhosh\\Desktop\\modis_geotif")
    
    }
    else{
      setwd("C:\\Users\\Santhosh\\Desktop\\modis_averae")
      writeRaster(STACK1, filename =paste(dates[i],".tif",sep=""), driver = "GeoTiff")
      setwd("C:\\Users\\Santhosh\\Desktop\\modis_geotif")
      
    }
    k=1
    s=list()
  }
}
s[k]=f[i]
resultingStack <- stack()
for(j in 1:length(s)){
  print(j)
  tempraster <- crop(raster(s[[j]]),a)
  resultingStack <- stack(resultingStack,tempraster)
}
mean2 <- calc(resultingStack, fun = mean, na.rm = T)
# setwd("C:\\Users\\Santhosh\\Desktop\\result_aod")
setwd("C:\\Users\\Santhosh\\Desktop\\modis_averae")
writeRaster(x = mean2, filename =paste(dates[i],".tif",sep=""), driver = "GeoTiff")
# setwd("C:\\Users\\Santhosh\\Desktop\\test")
setwd("C:\\Users\\Santhosh\\Desktop\\modis_geotif")






STACK1 <- stack(s)
mean2 <- calc(STACK1, fun = mean, na.rm = T)
setwd("C:\\Users\\Santhosh\\Desktop\\result_aod")
writeRaster(x = mean2, filename =paste(dates[i],".tif",sep=""), driver = "GeoTiff")
setwd("C:\\Users\\Santhosh\\Desktop\\test")

f <- f[with(f,order(as.POSIXct(substring(f,7,15),"%d%b%y")))]

# STACK1 <- stack(f)
# mean <- calc(STACK1, fun = mean, na.rm = T)
# writeRaster(x = mean, filename = "mean.tif", driver = "GeoTiff")
