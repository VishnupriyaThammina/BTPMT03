library(gdalUtils)
library(raster)
library(rgdal)

gdalinfo("C:\\Users\\Santhosh\\Desktop\\modis\\MOD04_L2.A2020012.0345.061.2020014192028.hdf")
sds <- get_subdatasets("C:\\Users\\Santhosh\\Desktop\\modis\\MOD04_L2.A2020012.0345.061.2020014192028.hdf")


sds <- get_subdatasets("D:\\HONOURS\\modis_3km\\MOD04_3K.A2020012.0335.061.2020014191822.hdf")

filename <- substr("MOD04_L2.A2020012.0345.061.2020014192028.hdf",11,22)
filename <- paste0("C:\\Users\\Santhosh\\Desktop\\modis_geotif\\",filename, ".tif")
gdal_translate(sds[60], dst_dataset =filename)

r<-raster("C:\\Users\\Santhosh\\Desktop\\modis_geotif\\2020012.0345.tif")
temp <- as.data.frame(r,xy=T)
