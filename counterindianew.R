library(sp)
library(spatstat)
library(raster)
library(maptools)
library(rgdal)
library(ggplot2)
library(rgeos)
library(sf)
library(plotly)
library(reshape2)
# to convert to point pattern

# nc <- sf::st_read("D:\\Downloads\\Downloads\\html_files\\HEATMAPS-20210424T120029Z-001\\2020\\IND_adm1.shp")
# nc <- sf::st_read(system.file("D:/Downloads/Downloads/html_files/HEATMAPS-20210424T120029Z-001/2020/IND_adm1.shp", package = "sf"), quiet = TRUE

data.shape<-readOGR(dsn="D:\\Downloads\\Downloads\\html_files\\HEATMAPS-20210424T120029Z-001\\2020\\IND_adm1.shp")
plot(data.shape)
df1 <- fortify(data.shape)



new=read.csv("D:\\HONOURS PROJECT\\RESULTS\\GRAPHS\\PREDICTED_24\\ground_24.csv",header = TRUE)
new<-data.frame(new)
new1<-cbind(as.numeric(new$longitude),as.numeric(new$latitude),as.numeric(new$mae),as.numeric(new$rmse),as.numeric(new$rscore))
colnames(new1)<-c("Lon","Lat","MAE","RMSE","Rscore")

new2<-cbind(as.numeric(new$longitude),as.numeric(new$latitude),as.numeric(new$mae))
colnames(new2)<-c("Lon","Lat","MAE")

out <- matrix(0, nrow=length(unique(new$longitude)), ncol=length(unique(new$latitude)))
out[cbind(new$longitude, new$latitude)] <- new$mae

df <- melt(out)

p<-geom_polygon(data=data.shape,aes(x=long,y=lat,group=group), colour="black",fill=NA)
  
p <-ggplot(df, aes(Var1,Var2, z= value)) +
  stat_contour(geom="polygon",aes(fill=stat(level))) +
  scale_fill_distiller(palette = "YlGn", direction = -1)


fig1<-ggplotly(p)
# fig2 <- geom_polygon(data=data.shape,aes(x=long,y=lat,group=group), colour="black",fill=NA)
fig2<-ggplot(df1, aes(long, lat, group = group),color="black") + 
  geom_polygon() 
  
subplot(fig1,fig2)

# library(maptools) 
# library(rgdal) 
# library(sp) 
# library(maptools) 
# library(sm) 
# require(akima) 
# require(spplot) 
# library(raster) 
# library(rgeos)
# 
# new=read.csv("D:\\HONOURS PROJECT\\RESULTS\\GRAPHS\\PREDICTED_24\\ground_24.csv",header = TRUE)
# new<-data.frame(new)
# x<-new$longitude
# y<-new$latitude
# z<-new$mae
# 
# fld <- interp(x,y,z)
# 
# par(mar=c(5,5,1,1)) filled.contour(fld)
# melt(volcano)
# 
# new1<-cbind(as.numeric(new$longitude),as.numeric(new$latitude),as.numeric(new$mae),as.numeric(new$rmse),as.numeric(new$rscore))
# colnames(new1)<-c("Lon","Lat","MAE","RMSE","Rscore")
 

