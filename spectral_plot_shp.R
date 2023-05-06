library(rgdal)
library(ggplot2)
library(rgeos)
library(sf)
library(plotly)
suppressMessages(library(spatstat))

# data.shape<-readOGR(dsn="D:\\Downloads\\Downloads\\html_files\\HEATMAPS-20210424T120029Z-001\\2020\\IND_adm1.shp")
data.shape<-readOGR(dsn="D:\\Downloads\\Downloads\\web series\\honors\\India_State_Shapefile\\India_State_Shapefile\\India_State_Boundary.shp")
# obs_window <- owin(data.shape@bbox[1,], data.shape@bbox[2,])

plot(data.shape)
# new=read.csv("D:\\HONOURS PROJECT\\RESULTS\\GRAPHS\\PREDICTED_24\\spectral_24.csv",header = TRUE)
new=read.csv("D:\\HONOURS PROJECT\\RESULTS\\GRAPHS\\PREDICTED_16\\spectral_16.csv",header = TRUE)
# new=read.csv("D:\\HONOURS PROJECT\\RESULTS\\GRAPHS\\PREDICTED_8\\spectral_8.csv",header = TRUE)

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
# 
# fig <- plot_ly(x=st$V2,y=st$V1,z=st$u, type = "contour")
# fig
# colfunc<-colorRampPalette(c("royalblue","green","yellow","red"))


# map<-ggplot(st,aes(V2,V1))+geom_raster(aes(fill=u))+geom_polygon(data=data.shape,aes(x=long,y=lat,group=group),color="black",fill=NA)+scale_fill_gradientn(name = "RMSE",colours=nickColors(12),space = "Lab",limits=c(10,32))
# map<-ggplot(st,aes(V2,V1))+geom_raster(aes(fill=u))+geom_polygon(data=data.shape,aes(x=long,y=lat,group=group),color="black",fill=NA)+scale_fill_gradientn(name = "RMSE",colours=brewer.pal(8,"Dark2"),space = "Lab",limits=c(10,32))

mybreaks <- seq(9,34, length.out =15)
map<-ggplot(st,aes(V2,V1,z=u))+geom_contour_filled(breaks = mybreaks)+ylab("Latitude")+xlab("Longitude")
map

# +geom_polygon(data=data.shape,aes(x=long,y=lat,group=group),color="black",fill=NA)
# 
# map1<-ggplot()+geom_polygon(data=data.shape,aes(x=long,y=lat,group=group),color="black",fill=NA)
# 
# coordinates(st)<-~V2+V1
# proj4string(st)<-proj4string(data.shape)
# z1<-over(st,data.shape)
# 
# map<-ggplot(st,aes(V2,V1,fill=u))+geom_polygon(data=data.shape,aes(x=long,y=lat,group=group),color="black",fill=NA)+geom_raster(aes(fill =u)) +scale_fill_continuous(type="viridis",limits=c(0,76))



# map
# nickColors <- function(n, h = c(120,400), l = c(.40,.70), s = c(.8,1), alpha = 1){
#   require(colorspace)
#   require(scales)
#   return (alpha(hex(HLS(seq(h[1],h[2],length.out = n), seq(l[1],l[2],length.out = n), seq(s[1],s[2],length.out=n))), alpha))
# }

# write.csv(st,"C:\\Users\\Santhosh\\Desktop\\bye\\st.csv", row.names = FALSE)


# ppp_malaria<-ppp(new1$Lon,new1$Lat,
#                  marks=new1$Rscore,window=obs_window)
# idw_malaria <- idw(ppp_malaria, power=1, at="pixels")
# rslt <- idw_malaria[data.shape, drop = FALSE]

# ggplot(idw_malaria,
#        col=terrain.colors(10)
# )+geom_polygon(data=data.shape,aes(x=long,y=lat,group=group), colour="black")
#   plot(idw_malaria,
#        col=terrain.colors(10), 
#        main="IDW method \n (Power = 0.05)")
# 
# 
# 
# map<-ggplot(new1, aes(Lon,Lat))+geom_polygon(data=data.shape,aes(x=long,y=lat,group=group), colour="black",fill=NA)+geom_point(aes(color=Rscore),size=2.5) +
#   scale_colour_gradient2(name = "RSCORE",limits=c(-1,1),low="blue",mid="green",high="red",breaks=c(0,0.3,0.6,0.9),space="Lab",guide = "colourbar")
# map
# 

