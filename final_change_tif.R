library(gdalUtils)
library(raster)
library(rgdal)
setwd("C:\\Users\\Santhosh\\Desktop\\modis")
files <- dir(pattern = ".hdf")
filename <- substr(files,11,22)
filename <- paste0("C:\\Users\\Santhosh\\Desktop\\modis_geotif\\",filename, ".tif")

i <- 1

for (i in 1:59){
  sds <- get_subdatasets(files[i])
  gdal_translate(sds[64], dst_dataset = filename[i])
}

