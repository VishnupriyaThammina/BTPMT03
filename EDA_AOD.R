library(rasterVis)
library(ggplot2)

s <- brick("C:\\Users\\Santhosh\\Desktop\\New folder\\2019\\oct\\3DIMG_01OCT2019_0530_L2G_AOD_AOD.tif")
s
crs(s)
xres(s)
yres(s)
res(s)
ncell(s)
dim(s)
 q <- raster("C:\\Users\\Santhosh\\Desktop\\New folder\\2019\\oct\\3DIMG_01OCT2019_0530_L2G_AOD_AOD.tif")
q 
