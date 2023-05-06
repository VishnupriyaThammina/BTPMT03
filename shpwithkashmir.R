library(rgdal)
library(ggplot2)
library(rgeos)
library(sf)
library(broom)

data.shape<-readShapeSpatial("D:\\Downloads\\Downloads\\web series\\honors\\India_State_Shapefile\\India_State_Shapefile\\India_State_Boundary.shp")
fortified <- tidy(data.shape,region="State_Name")


z1 <- read.csv("C:\\Users\\Santhosh\\Desktop\\bye\\spectral.csv")
z2<-data.frame(cbind(z1$longitude,z1$latitude,z1$pm25))
names(z2)<-c("lon","lat","pm")
map <- ggplot()+geom_point(data = z2, aes(x = lon, y = lat), size = 1.5,color="blue")+ geom_polygon(data = fortified, aes(x = long, y = lat, group = group),
                               fill="orange2",colour = "black",size=0.5)

map<-ggplot()+geom_polygon(data = data.shape, aes(x = long, y = lat, group = group),
            fill="white",colour = "black",size=0.5)+geom_point(data = z2, aes(x = lon, y = lat), size = 3,color="blue",shape='+')
map +theme_bw()


shapef<-readOGR("D:\\Downloads\\Downloads\\web series\\honors\\India_State_Shapefile\\India_State_Shapefile\\India_State_Boundary.shp")