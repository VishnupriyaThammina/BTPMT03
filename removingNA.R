library(rgdal)
library("readxl")
library(dplyr)
library(tidyverse)


df<-read.csv("D:\\HONOURS PROJECT\\DATASET\\cleanded_landsat8_data\\delhi_landsat8_station_data.csv")

s<-read.csv("D:\\HONOURS PROJECT\\DATASET\\cleanded_landsat8_data\\datewise_avg.csv")
colnames(s)<-c("id","dates","pm25")

df2<-merge(df,s,by=c("dates","id"),all=TRUE)
