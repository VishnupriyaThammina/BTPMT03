library(raster)
library(rgdal)

shp <- readOGR("C:\\Users\\Santhosh\\Downloads\\newshape\\in\\in.shp")     

infiles <- list.files(path=getwd(), 
                      pattern="*.tif", 
                      full.names=TRUE)
outfiles<-dir(pattern = ".tif")

for (i in seq_along(infiles)) {
  r <- crop(raster(infiles[i]), shp)
  setwd("C:/Users/Santhosh/Desktop/modis_india")
  writeRaster(r, filename=outfiles[i])
  setwd("C:/Users/Santhosh/Desktop/modis_geotif")
}
