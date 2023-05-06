library("readxl")
library("openxlsx")
library("dplyr")

location <- read.csv("D:\\HONOURS PROJECT\\Grid_Delhi_1691.csv")
location1 <- read.csv("C:\\Users\\Santhosh\\Pictures\\dip\\scatterplots\\braine\\dip\\monitoringStn_PM25.csv")

wanted <- c()
for(i in 1:2094){
  for(j in 1:258){
    if((location$x1[i]==location1$longitude[j])&&(location$x2[i]==location1$latitude[j])){
      wanted<-append(wanted,i)
    }
  }
}
