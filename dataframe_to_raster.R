library(raster)
# create spatial points data frame
spg <- df
coordinates(spg) <- ~ x + y
# coerce to SpatialPixelsDataFrame
gridded(spg) <- TRUE
# coerce to raster
rasterDF <- raster(spg)
rasterDF
