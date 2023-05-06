library("readxl")
library("openxlsx")
library("dplyr")
library("ggplot2")

india <- read.csv("C:\\Users\\Santhosh\\Pictures\\dip\\scatterplots\\braine\\dip\\stations.csv")
groundindia <- read.csv("C:\\Users\\Santhosh\\Pictures\\dip\\scatterplots\\braine\\dip\\sgs_santosh\\ground_new.csv")
ggplot()+geom_tile(data=groundindia,aes(x=x1,y=x2,fill=PM25))+theme_bw()

hist(india$PM25)
