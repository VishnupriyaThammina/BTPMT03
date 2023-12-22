imageData = Tiff('PRS_L2C_STD_20211004053414_20211004053418_0001_HCO_FULL.tif')
ss=read(imageData)
new=ss(:,:,2:37)
countNA=[]
for i= 1:36
  z=ss(:,:,i)
  zz=reshape(z,1,[]);
  a=sum(zz==-999.0000)
  countNA=[countNA; a]
  %fprintf('%d',i)
end
1213*1194
info = geotiffinfo('PRS_L2C_STD_20211004053414_20211004053418_0001_HCO_FULL.tif');
[x,y] = pix2map(info.RefMatrix, 1, 1);
[lat,lon] = projinv(info, x,y)
height = info.Height; % Integer indicating the height of the image in pixels
width = info.Width; % Integer indicating the width of the image in pixels
[cols,rows] = meshgrid(1:width,1:height);
[x,y] = pix2map(info.RefMatrix, rows, cols);
[lat,lon] = projinv(info, x,y);
[ADlat,ADlon] = pix2latlon(info.RefMatrix, rows, cols);

for i =1:36
  z=ss(:,:,i)
  zz=reshape(z,1,[]);
  a=zz(~(zz==-999.0000))
  h=histogram(a)
  xlabel('Values')
  ylabel('Count')
  title(['Band no.',sprintf('%d',i)])
  %imhist(a)
  saveas(gcf,sprintf('HIS%d.png',i))
  %imwrite(h,sprintf('%d.jpg',i))
  %saveas(h,sprintf('FIG%d.png',i));
end
minm=[]
maxm=[]
avgm=[]
mednm=[]
stdm=[]
varm=[]
for i =1:36
  z=ss(:,:,i)
  zz=reshape(z,1,[]);
  a=zz(~(zz==-999.0000)) 
  mn=min(a)
  mx=max(a)
  av=mean(a)
  md=median(a)
  sd=std(a)
  vr=var(a)
  minm=[minm;mn]
  maxm=[maxm;mx]
  avgm=[avgm;av]
  mednm=[mednm;md]
  stdm=[stdm;sd]
  varm=[varm;vr]
end
xx=reshape(x,1,[])
yy=reshape(y,1,[])
zz=reshape(z,1,[])
xxx=xx(~(zz==-999.0000))
yyy=yy(~(zz==-999.0000))
zzz=zz(~(zz==-999.0000))
zzz=reshape(zzz,1,1040531);
  surf(x,y,z)
  contour(zzz)
%%%%%%%%
coords=xlsread('Delhi_Stations.xlsx')
stlats=coords(:,3)
stlons=coords(:,2)
s=length(stlats)
xxx
%for i=1:s
  %xxx==stlats(i) stlons(i)
  
[x,y] = pix2map(info.SpatialRef,  rows, cols)  
R = info.SpatialRef;[x,y] = R.intrinsicToWorld(1,1);
[lat,lon] = projinv(info, x,y);

  
I = imread('PRS_L2C_STD_20211004053414_20211004053418_0001_HCO_FULL.tif');
info = geotiffinfo('PRS_L2C_STD_20211004053414_20211004053418_0001_HCO_FULL.tif');
Latlon = xlsread('monitoring_Stations.xlsx');
height = info.Height; % Integer indicating the height of the image in pixels
width = info.Width; % Integer indicating the width of the image in pixels
[rows,cols] = meshgrid(1:height,1:width);
[ADlat,ADlon] = pix2latlon(info.RefMatrix, rows, cols);
Latlon =readtable('monitoring_stn_bandval.csv')

Latlon=table2array(Latlon)
stlons=Latlon{:,1}
stlons=table2array(Latlon(:,3))
stlats=table2array(Latlon(:,2))
mask = I == (intmin('int32')+1);
pixval=[]
for i=2:37
   Idub = double(I(:,:,i));
   Idub(mask) = nan;
   pixelvalues = interp2(ADlat, ADlon, Idub,stlats, stlons); 
   pixval=[pixval, pixelvalues]
end


for j=2:37
  Idub = double(I(:,:,j));
  Idub=reshape(Idub,1,[])
  Values=[]
  Lats=[]
  Lons=[]
  zz=Idub
  xxx=xx(~(zz==-999.0000))
  yyy=yy(~(zz==-999.0000))
  zzz=zz(~(zz==-999.0000))
  for i=1:38
    idx = knnsearch([yyy.' xxx.'],[stlons(i) stlats(i)],'K',100);
    xxx1=xxx.'
    yyy1=yyy.'
    idx=idx.'
    nearest_lat = xxx1(idx);
    nearest_lon = yyy1(idx);
    nearest_Avalue = zzz(idx);   
    nearest_Avalue = nearest_Avalue'   
    Values=[Values nearest_Avalue]
    Lats=[Lats nearest_lat]
    Lons=[Lons nearest_lon ]
  end
  filename = sprintf('BandValues%d.csv',j);
  csvwrite(filename,Values)
  filename1 = sprintf('Lats%d.csv',j);
  csvwrite(filename1,Lats)
  filename2 = sprintf('Lons%d.csv',j);
  csvwrite(filename2,Lons)
end



xx=reshape(ADlat,1,[])
yy=reshape(ADlon,1,[])
xx=xx.'
yy=yy.'
Lats=[]
Lons=[]
for i=1:38
    idx = knnsearch([yy xx],[stlons(i) stlats(i)],'K',120);
    idx=idx.'
    nearest_lat = xx(idx);
    nearest_lon = yy(idx);
    filename1 = sprintf('NearestLatLonsStnID%d.csv',i);
    csvwrite(filename1,[nearest_lat  nearest_lon])
    Lats=[Lats nearest_lat]
    Lons=[Lons nearest_lon]
end
Lats=reshape(Lats,1,[])
Lons=reshape(Lons,1,[])
Lats=Lats.'
Lons=Lons.'
csvwrite('NearestNbrsLatLonAllStns.csv',[Lats Lons])
  