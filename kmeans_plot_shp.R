library(rgdal)
library(ggplot2)
library(rgeos)
library(sf)
# data.shape<-readOGR(dsn="D:\\Downloads\\Downloads\\html_files\\HEATMAPS-20210424T120029Z-001\\2020\\IND_adm1.shp")
data.shape<-readOGR(dsn="D:\\Downloads\\Downloads\\web series\\honors\\India_State_Shapefile\\India_State_Shapefile\\India_State_Boundary.shp")

plot(data.shape)
# new=read.csv("D:\\HONOURS PROJECT\\RESULTS\\GRAPHS\\PREDICTED_24\\kmeans_24.csv",header = TRUE)
new=read.csv("D:\\HONOURS PROJECT\\RESULTS\\GRAPHS\\PREDICTED_16\\K_means_16.csv",header = TRUE)
# new=read.csv("D:\\HONOURS PROJECT\\RESULTS\\GRAPHS\\PREDICTED_8\\K_means_8.csv",header = TRUE)

new<-data.frame(new)
new1<-cbind(as.numeric(new$longitude),as.numeric(new$latitude),as.numeric(new$rscore),as.numeric(new$mae),as.numeric(new$rmse))
colnames(new1)<-c("Lon","Lat","Rscore","MAE","RMSE")
new1<-data.frame(new1)
x<- new1$Lat
y<-new1$Lon
z<-new1$RMSE
xi<-grid1$Lat
yi<-grid1$Lon
st<-data.frame(idwr(x,y,z,xi,yi))

fig <- plot_ly(x=st$V2,y=st$V1,z=st$u, type = "contour")
fig


map<-ggplot(st,aes(V2,V1))+geom_raster(aes(fill=u))+geom_polygon(data=data.shape,aes(x=long,y=lat,group=group),color="black",fill=NA)+scale_fill_gradientn(name = "RMSE",colours=terrain.colors(10),breaks=c(24,28,30))
# map<-ggplot(st,aes(V2,V1))+geom_raster(aes(fill=u))+geom_polygon(data=data.shape,aes(x=long,y=lat,group=group),color="black",fill=NA)+scale_fill_gradientn(name = "RMSE",colours=terrain.colors(10))
map


# map<-ggplot(new1, aes(Lon,Lat))+geom_polygon(data=data.shape,aes(x=long,y=lat,group=group), colour="black",fill=NA)+geom_point(aes(color=Rscore),size=3) +
#   scale_colour_gradient2(name = "RSCORE",limits=c(-1,1),low="blue",mid="green",high="red",breaks=c(0,0.3,0.6,0.9),space="Lab",guide = "colourbar")
#   # ggtitle("Weighted K-Means")+
#   # theme(plot.title = element_text(hjust = 0.5))
# map

