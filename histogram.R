library("readxl")
library("lubridate")
library("zoo")
library("dplyr")
require(ggplot2)
library(data.table)
library(lubridate)
library(RchivalTag)
library(stringr)
library(janitor)

library(data.table)
library(tidyverse)

# 
# data <-read.csv("D:\\HONOURS PROJECT\\DATASET\\cleanded_landsat8_data\\delhi_landsat8_station_data.csv",header=TRUE)
# 
# 
# hist(data$B10, breaks = 1000, col = "green")
# 
# z<-read_xlsx("D:\\Downloads\\Downloads\\web series\\honors\\landsat8\\delhi_30min\\09-01-2020_cpcb.xlsx")







dataorganize<-function(data){
  l<-data.frame(cbind(data[,1],data[,2]))
  s1<-which(is.na(l[,1]))
  s2<-which(is.na(l[,2]))
  common <- intersect(s1, s2)
  len<-common[end(common)][1]-common[length(common)-1][1]
  ss=c(1:len)
  for (i in common[c(-1)]){
    # print(pm253[i,2:8])
    print(i)
    if (i== common[c(-1)][1]){
      date<- data[i:(i+len-1),1]
      ss<-cbind(ss,date)
    }
    L<-data[c(-1,-2)][i:(i+len-1),]
    ss<-cbind(ss,L)
  } 
  ss<-ss[c(-1)]
  ss<-remove_empty(ss)
  colnames(ss)<-ss[1,]
  z<-colnames(ss)
  # r<-z[1]
  colnames(ss)[1]<-"date"
  ss=ss[-1,]
  ss=ss[-1,]
  # ss<-rename(ss,c("r"="date"))
  return(ss)
}


setwd("D:\\Downloads\\Downloads\\web series\\honors\\landsat8\\delhi_30min")
temp <- list.files(pattern=".xlsx")
output <- c()
for(i in 1:length(temp)){
  u <- read_xlsx(temp[i])
  k<-dataorganize(u)
  output<-rbind(output,k[1,])
}
write.csv(output,"D:\\HONOURS PROJECT\\DATASET\\lat_lon\\delhi_needed.csv")


delhi<-read.csv("D:\\HONOURS PROJECT\\DATASET\\lat_lon\\delhi_needed.csv")
delhi_pm25<-delhi[,1]
delhi_pm10<-delhi[,1]
o<-names(delhi)
delhi_names<-o[1]
for(i in seq(2,81,2)){
  delhi_pm25<-cbind(delhi_pm25,delhi[,i])
  delhi_pm10<-cbind(delhi_pm10,delhi[,i+1])
  delhi_names <- cbind(delhi_names,o[i])
}

write.csv(delhi_pm10,"D:\\HONOURS PROJECT\\DATASET\\lat_lon\\delhi_pm10.csv")
write.csv(delhi_pm25,"D:\\HONOURS PROJECT\\DATASET\\lat_lon\\delhi_pm25.csv")
