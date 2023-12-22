setwd("D:\\AllGroun")
library(ncdf4)
library(chron)
library(readxl)
library(stringr)
library(dplyr)

setwd("D:\\AllGroun\\cleaned data")
x <- read_excel("D:\\AllGroun\\latitude_longitude.xlsx")
y <- read_excel("wsfinal.xlsx")

uv=nc_open("C3S-LC-L4-LCCS-Map-300m-P1Y-2019-v2.1.1.nc")
{
  sink('gimms3g_ndvi_1982-2012_metadata.txt')
  print(uv)
  sink()
}

# LonIdx <- which( uv$dim$lon$vals > 6.44 | uv$dim$lon$vals < 35.30)
# LatIdx <- which( uv$dim$lat$vals > 68.7 & uv$dim$lat$vals < 97.25)
MyVariable <- ncvar_get( uv,"blh")[ LonIdx, LatIdx]

lat_data <- read_excel("states_latitude_longitude.xlsx")

lat_data %>% 
  mutate(across(where(is.character), str_remove_all, pattern = fixed(" ")))


lon <- ncvar_get(nc_data, "longitude")
lat <- ncvar_get(nc_data, "latitude", verbose = F)
lat_rng = c(5,25)
lon_rng=c(80,100)
t <- ncvar_get(uv, "time")
tunits <- ncatt_get(uv,"time","units")
ndvi.array <- ncvar_get(uv, "blh") 

tustr <- strsplit(tunits$value, " ")
tdstr <- strsplit(unlist(tustr)[3], "-")
tmonth <- as.integer(unlist(tdstr)[2])
tday <- as.integer(substring(unlist(tdstr)[3],1,2))
tyear <- as.integer(unlist(tdstr)[1])
time.val=as.Date(chron(t,origin=c(tmonth, tday, tyear)))

r_brick <- brick(ndvi.array,xmn=min(lat), xmx=max(lat), ymn=min(lon), ymx=max(lon), crs=CRS("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs+ towgs84=0,0,0"))

extract(r_brick, SpatialPoints(cbind(91.78,26.181)), method='simple')


x<-c()
p<-c()
for(i in 1:length(lat_data$latitude)){
  toolik_lat<-as.numeric(str_extract(lat_data$latitude[i],"\\d+.\\d+"))
  toolik_lon<-as.numeric(str_extract(lat_data$longitude[i],"\\d+.\\d+"))
  toolik_series <- extract(r_brick, SpatialPoints(cbind(toolik_lon,toolik_lat)), method='simple')
  x<-rbind(x,toolik_series)
  print(x)
  p<-cbind(p,lat_data$station_name[i])
}
y<-t(x)


getNcTime <- function(nc) {
  require(lubridate)
  ncdims <- names(nc$dim) #get netcdf dimensions
  timevar <- ncdims[which(ncdims %in% c("time", "Time", "datetime", "Datetime", "date", "Date"))[1]] #find time variable
  times <- ncvar_get(nc, timevar)
  if (length(timevar)==0) stop("ERROR! Could not identify the correct time variable")
  timeatt <- ncatt_get(nc, timevar) #get attributes
  timedef <- strsplit(timeatt$units, " ")[[1]]
  timeunit <- timedef[1]
  tz <- timedef[5]
  timestart <- strsplit(timedef[4], ":")[[1]]
  if (length(timestart) != 3 || timestart[1] > 24 || timestart[2] > 60 || timestart[3] > 60 || any(timestart < 0)) {
    cat("Warning:", timestart, "not a valid start time. Assuming 00:00:00\n")
    warning(paste("Warning:", timestart, "not a valid start time. Assuming 00:00:00\n"))
    timedef[4] <- "00:00:00"
  }
  if (! tz %in% OlsonNames()) {
    cat("Warning:", tz, "not a valid timezone. Assuming UTC\n")
    warning(paste("Warning:", timestart, "not a valid start time. Assuming 00:00:00\n"))
    tz <- "UTC"
  }
  timestart <- ymd_hms(paste(timedef[3], timedef[4]), tz=tz)
  f <- switch(tolower(timeunit), #Find the correct lubridate time function based on the unit
              seconds=seconds, second=seconds, sec=seconds,
              minutes=minutes, minute=minutes, min=minutes,
              hours=hours,     hour=hours,     h=hours,
              days=days,       day=days,       d=days,
              months=months,   month=months,   m=months,
              years=years,     year=years,     yr=years,
              NA
  )
  suppressWarnings(if (is.na(f)) stop("Could not understand the time unit format"))
  timestart + f(times)
}

