library(raster)
library(ggplot2)
library(animation)

r<-raster("C:\\Users\\Santhosh\\Desktop\\modis_india\\2020012.tif")
r1<-raster("C:\\Users\\Santhosh\\Desktop\\modis_india\\2020013.tif")
r2<-raster("C:\\Users\\Santhosh\\Desktop\\modis_india\\2020014.tif")
r3<-raster("C:\\Users\\Santhosh\\Desktop\\modis_india\\2020015.tif")
r4<-raster("C:\\Users\\Santhosh\\Desktop\\modis_india\\2020016.tif")
r5<-raster("C:\\Users\\Santhosh\\Desktop\\modis_india\\2020017.tif")
r6<-raster("C:\\Users\\Santhosh\\Desktop\\modis_india\\2020018.tif")
# r7<-raster("C:\\Users\\Santhosh\\Desktop\\modis_averae\\2020012.tif")


k<-c(r,r1,r2,r3,r4,r5,r6)

saveGIF({
  for (i in k) {
    # temp <- as.data.frame(i,xy=T)
    temp<-as.data.frame(rasterToPoints(i))
    breaks <-c("0.1","0.3","0.7","1","2","3","4")
    p = ggplot(temp, aes(x = x, y = y,fill=temp[,3]))+
      geom_raster() +scale_fill_gradientn(name=colnames(temp)[3],colours =c("orange","blue","green","yellow","orange"),breaks=c(0.15,0.30,0.45,1,1.5),na.value = "grey50",limits=c(0,1.5))
    
    print(p)
  }
}, movie.name="C:\\Users\\Santhosh\\Pictures\\Endsem\\test.gif")

