library(readxl)
library(gdalUtils)
library(raster)
library(rgdal)

data <- read_excel("D:\\Downloads\\Downloads\\web series\\honors\\Delhi_Stations.xlsx")
sds <- get_subdatasets("D:\\HONOURS PROJECT\\DATASET\\MODIS_Delhi\\~Downloads\\MODIS\\MOD04_3K.A2019002.0510.061.2019009001535.hdf")
filename <- substr("MOD04_3K.A2019001.0605.061.2019009000509.hdf",11,22)
filename <- paste0("D:\\HONOURS PROJECT\\DATASET\\MODIS_Delhi\\GeoTiff\\",filename, ".tif")
gdal_translate(sds[11], dst_dataset =filename,crs = "")

raster_file = raster("D:\\HONOURS PROJECT\\DATASET\\MODIS_Delhi\\GeoTiff\\2019001.0605.tif")
points         <- cbind(data$Longitude,data$Latitude)
out   <- extract(raster_file, points)



gdal_translate -of GTiff -sds filename "xyz.tiff"


