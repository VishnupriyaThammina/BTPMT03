library(rgdal)
library(ggplot2)
library(rgeos)
library(sf)
data.shape<-readOGR(dsn="D:\\Downloads\\Downloads\\html_files\\HEATMAPS-20210424T120029Z-001\\2020\\IND_adm1.shp")
plot(data.shape)
new=read.csv("D:\\Downloads\\Downloads\\web series\\honors\\windmean.csv",header = TRUE)

new<-data.frame(new)
new1<-cbind(as.numeric(new$longitude),as.numeric(new$latitude),as.numeric(new$windspeedmean),as.numeric(new$windspeedstd),as.numeric(new$winddirectionmean),as.numeric(new$winddirectionstd))
colnames(new1)<-c("Lon","Lat","windspeedmean","windspeedstd","winddirectionmean","winddirectionstd")
new1<-data.frame(new1)
map<-ggplot(new1, aes(Lon,Lat))+geom_polygon(data=data.shape,aes(x=long,y=lat,group=group), colour="black",fill=NA)+geom_point(aes(color=winddirectionstd),size=3) +
  scale_colour_gradient2(name = "Winddirection Std",limits=c(20,110),low="blue",mid="green",high="red",breaks=c(25,50,70,90),space="Lab",guide = "colourbar")
map

