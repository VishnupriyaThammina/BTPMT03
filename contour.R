library(rgdal)
library(ggplot2)
library(rgeos)
library(sf)
library(plotly)
data.shape<-readOGR(dsn="D:\\Downloads\\Downloads\\html_files\\HEATMAPS-20210424T120029Z-001\\2020\\IND_adm1.shp")
plot(data.shape)
new=read.csv("D:\\HONOURS PROJECT\\RESULTS\\GRAPHS\\PREDICTED_24\\ground_24.csv",header = TRUE)

new<-data.frame(new)
new1<-cbind(as.numeric(new$longitude),as.numeric(new$latitude),as.numeric(new$mae),as.numeric(new$rmse),as.numeric(new$rscore))
colnames(new1)<-c("Lon","Lat","MAE","RMSE","Rscore")

new1<-data.frame(new1)
map<-ggplot(new1, aes(Lon,Lat))+geom_polygon(data=data.shape,aes(x=long,y=lat,group=group), colour="black",fill=NA)+
  # plot_ly(x=new1$Lon,y=new1$Lat,z=new1$MAE,type="contour")
  # geom_density2d(data=new1,aes(x=Lon,y=Lat),colour="red")

  geom_point(data = new1, mapping = aes(Lon, Lat, colour=MAE), size=4, alpha=0.5, shape=15) + scale_colour_gradient(low = 'green', high = 'red')

  # stat_density2d(data = new1, mapping = aes(x = Lon, y = Lat, fill=MAE), geom = "polygon", alpha = 0.1, contour = TRUE)

map

