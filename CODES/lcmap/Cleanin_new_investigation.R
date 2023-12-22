
gridpts<-read.csv('/home/subhojit/Downloads/Data_Analysis_Prisma/Dataset_V2/Spatial_Estimation/gridpoints.csv',header=FALSE)
names(gridpts)<-c("Lon","Lat")
coordinates(points)<-~Lat+Lon
out1<-c()
for(i in 2:37){
  band<- r[[i]]
  out<-extract(band, gridpts)
  names(out)<- band
  out1<-cbind(out1,out)
  #out1<-data.frame(test.dat$latitude, test.dat$longitude, out)
  #form = sprintf('band_%s.csv', i)
  #write.csv(out1,file=form)
}
out1<-data.frame(out1)
write.csv(out1, 'gridpoints_feat.csv')
ind_exclude<-c()
for (i in 1:1448322){
  #r2<-out1[i,]
  indx <- apply(out1[i,], 2, function(x) any(is.na(x) | is.infinite(x))) 
  if (length(indx[indx==TRUE])==36){
  ind_exclude<- rbind(ind_exclude,TRUE)
  }
  else{
    ind_exclude<- rbind(ind_exclude,FALSE)
  }
}
ind_exclude<-data.frame(ind_exclude)
ind_exclude_n<-!ind_exclude
zlon<-gridpts$Lon[unlist(!ind_exclude)]
zlat<-gridpts$Lat[unlist(!ind_exclude)]
z<-data.frame(cbind(zlon,zlat))


map<-ggplot()+geom_polygon(data=data.shape,aes(x=long,y=lat,group=group), colour="black",fill=NA)+geom_point(data=z,mapping=aes(x=zlon,y=zlat), colour="red")
map+theme_void()
excluded<-c()
for (i in 1:36){
  #r2<-out1[i,]
  rs<-out1[,i][unlist(!ind_exclude)]
  excluded<-cbind(excluded,rs)
}
excluded<-data.frame(excluded)
### excluded1 contains after exclusion working points are taken
excluded1<-excluded

mean2 <- apply(excluded,2,function(x) mean(na.omit(x)))
for (i in 1:36){
  excluded1[is.na(excluded1[,i]),i]<-mean2[i]
}

excluded1<-cbind(excluded1,z)

mean(excluded[ , i], na.rm = TRUE)
for(i in 1:36) {
  gridpts[,i][is.na(gridpts[,i])] <- mean(gridpts[,i], na.rm = TRUE)
}
write.csv(gridpts,'Dataset2_Features_Cleaned_Spatial_EStimation.csv')

write.csv(excluded1,'gridpts_feat_cleaned.csv')
colnames(excluded1)<-c()









