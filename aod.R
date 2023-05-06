library(raster)
r <- raster("C:\\Users\\Santhosh\\Desktop\\result_aod\\2020-01-02.tif")
p<-rasterToPoints(r)
l<-nscore(p[,"X2020.01.02"])
hist(l$nscore,xlab = "Weight",col = "yellow",border = "blue")
hist(l$trn.table[,"x"],xlab = "Weight",col = "yellow",border = "blue")
library(ggplot2)

library(gganimate)

r1<-raster("C:\\Users\\Santhosh\\Desktop\\test\\3DIMG_01APR2019_0530_L2G_AOD_AOD.tif")
r2<-raster("C:\\Users\\Santhosh\\Desktop\\test\\3DIMG_01APR2019_0600_L2G_AOD_AOD.tif")
r3<-raster("C:\\Users\\Santhosh\\Desktop\\test\\3DIMG_01APR2019_0630_L2G_AOD_AOD.tif")
r4<-raster("C:\\Users\\Santhosh\\Desktop\\test\\3DIMG_01APR2019_0700_L2G_AOD_AOD.tif")
r5<-raster("C:\\Users\\Santhosh\\Desktop\\test\\3DIMG_01APR2019_0730_L2G_AOD_AOD.tif")
r6<-raster("C:\\Users\\Santhosh\\Desktop\\test\\3DIMG_01APR2019_0800_L2G_AOD_AOD.tif")
r7<-raster("C:\\Users\\Santhosh\\Desktop\\test\\3DIMG_01APR2019_0830_L2G_AOD_AOD.tif")


r8<-raster("C:\\Users\\Santhosh\\Pictures\\Endsem\\3DIMG_01APR2019_AVRG_L2G_AOD_AOD.tif")

k<-c(r1,r2,r3,r4,r5,r6,r7,r8)

temp <- as.data.frame(r1,xy=T)
temp1 <- as.data.frame(r2,xy=T)
temp2 <- as.data.frame(r3,xy=T)
temp3 <- as.data.frame(r4,xy=T)
temp4 <- as.data.frame(r5,xy=T)
temp5 <- as.data.frame(r6,xy=T)
temp6 <- as.data.frame(r7,xy=T)
temp7 <- as.data.frame(r8,xy=T)

ggplot(temp7, aes(x = x, y = y,color = "red",fill=temp7[,3]))+
  geom_raster()+scale_fill_gradientn(breaks=c(1,2,3,4),colors = c("white","blue","green","yellow"))

saveGIF({
  for (i in k) {
    temp <- as.data.frame(i,xy=T)
    p = ggplot(temp, aes(x = x, y = y,fill=temp[,3]))+
      geom_raster() +scale_fill_gradient(name=substring(colnames(temp)[3],8,21),breaks=c(1,2,3,4),limits=c(0,5))
    print(p)
  }
}, movie.name="C:\\Users\\Santhosh\\Pictures\\Endsem\\test.gif")
