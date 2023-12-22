# -*- coding: utf-8 -*-
"""k=3Delhi_Clustering_stations.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10kqHK1vAjrBVLKhL5Cbo6OHC34B6ojDy
"""

!pip install pyinform

import pandas as pd
import numpy as np
from numpy import genfromtxt

pm = genfromtxt('/content/drive/MyDrive/Delhi_SAGNN_files/Dataset_V4/eju_delhi_pm25.csv', delimiter=',')

pm

pm=pm[1:52536,1:]

pm

import pandas as pd
pm=pd.DataFrame(pm)

loc_a

loc_a

from math import radians, cos, sin, asin, sqrt
def dist(lat1, long1, lat2, long2):
    """
Replicating the same formula as mentioned in Wiki
    """
    # convert decimal degrees to radians 
    lat1, long1, lat2, long2 = map(radians, [lat1, long1, lat2, long2])
    # haversine formula 
    dlon = long2 - long1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km

def find_nearest(lat, long):
    distances = loc_b.apply(
        lambda row: dist(lat, long, row['lat'], row['lon']), 
        axis=1)
    return loc_b.loc[distances.idxmin(), 'stn_name']

loc_b

loc_b.apply(
    lambda row: find_nearest(row['lat'], row['lon']), axis=0)

loc_b['name'] = loc_b.apply(
    lambda row: find_nearest(row['lat'], row['lon']), 
    axis=1)# To check the data frame if it has a new column of hotel name (for each and every member's location in the list)members.head()

import pandas as pd
ar=pd.read_csv('/content/drive/MyDrive/Delhi_SAGNN_files/GNNDelhi/data/stn_coord.csv')
loc_a=pd.DataFrame(ar)
loc_b=pd.DataFrame(ar)
from sklearn.neighbors import BallTree, KDTree
import numpy as np
#bt = BallTree(np.deg2rad(loc_b[['lat', 'lon']].values), metric='haversine')
kd = KDTree(loc_b[['lat', 'lon']].values, metric = 'minkowski')
distances1, indices1 = kd.query(loc_b[['lat', 'lon']],k=7)
all_stn_ind=[]
for i in range(38):
  # print(Hour_3_sample[i,:,:].shape)
  j=indices1[i]
  print(j)
  all_stn_ind.append(np.expand_dims(j,0))
all_stn_ind=np.concatenate(all_stn_ind,axis=0)
  #each_stn=[]
  # for l in j:
  #   # print(l)
  #   # print(Hour_3_sample[:,l,:].shape)
  #   each_stn.append(ar['MAE'].iloc[[l]])
  # each_stn=np.concatenate(each_stn,axis=-1)
  # # print(each_stn.shape)
  # all_stn.append(each_stn)

a=pm.corr(method ='pearson')

a.to_csv("corr.csv")

all_stn_ind.shape

n=all_stn_ind[2][0]
n

n=all_stn_ind[0]
n

a[n].iloc[0]

n=all_stn_ind[0]
corrnew=a[all_stn_ind[0]]
corrnew.shape

corr_list=[]
for i in range(0,38):
  corrnew=a[all_stn_ind[i]].iloc[i]
  print(corrnew.shape)
  corr_list.append(np.expand_dims(corrnew,0))
corr_list=np.concatenate(corr_list,0)

corr_list.shape

corr_list=pd.DataFrame(corr_list)
corr_list.to_csv("St-wise_corr_list.csv")

from pyinform import mutual_info
# mutual_info(pm[:,0], pm[:,3])

corr=np.zeros((38,38))
for i in range(38):
  for j in range(38):
    corr[i,j]=mutual_info(pm[:,i], pm[:,j])

corr

import seaborn as sns
import pandas as pd
corr=pd.DataFrame(corr)
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform

X = [[i] for i in [2, 8, 0, 4, 1, 9, 9, 0]]

labels = fcluster(Z, 1.24, criterion='distance')
labels

Z

Z = linkage(corr, 'ward')

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(25, 10))
dn = dendrogram(Z)

import pandas as pd
from datetime import datetime
# h=pd.date_range(start="2018-01-01",end="2020-12-31", freq='3H').to_pydatetime().tolist()
# h=h[:8756]
# h=pd.DataFrame(h)
# import numpy as np
# sam=np.load("/content/drive/MyDrive/Delhi Lockdown/DelhiData_3hr_sample_V9_Eju.npy")
# pm=sam[:,:,0]
# # pm
ar=pd.read_csv('/content/drive/MyDrive/Delhi_SAGNN_files/GNNDelhi/data/stn_coord.csv')
loc_b=pd.DataFrame(ar)

ar



pm=pd.DataFrame(pm)
pm.columns=loc_b['stn_name']

pm.columns=loc_b['stn_name']

pm

a=np.linspace(1, 38,dtype = int, num=38)
a

pm=pd.DataFrame(pm)
pm.columns=a

pm

pm = pd.to_numeric(pm)

# pm=pm.T

co=pm.corr()

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#!pip install plotly
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
import numpy as np
# import datetime as dt
# from datetime import timedelta
# from sklearn.model_selection import GridSearchCV
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score,silhouette_samples
# from sklearn.linear_model import LinearRegression,Ridge,Lasso
# from sklearn.svm import SVR
# from sklearn.metrics import mean_squared_error,r2_score
import statsmodels.api as sm
# from statsmodels.tsa.api import Holt,SimpleExpSmoothing,ExponentialSmoothing
# from fbprophet import Prophet
# from sklearn.preprocessing import PolynomialFeatures
# from statsmodels.tsa.stattools import adfuller
fig, ax = plt.subplots(figsize=(16,10))
snsplot=sns.heatmap(co,ax=ax)
fig = snsplot.get_figure()
fig.savefig("/content/drive/MyDrive/Delhi_Model_Results/corr1.png")

fig.savefig("/content/drive/MyDrive/Delhi_Model_Results/corr.pdf")

from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=1000,n_init=10,random_state=32)
    kmeans.fit(pm)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
# plt.savefig("/content/drive/MyDrive/Delhi_Model_Results/elbow.pdf")
plt.show()

# kmeans = KMeans(n_clusters=3,init='k-means++',max_iter=300,n_init=10,random_state=0) 
# y_kmeans = kmeans.fit_predict(pm)

pm= pm.as_matrix(columns=None)

pm

def doKmeans(X, nclust=8):
    model = KMeans(nclust)
    model.fit(X)
    clust_labels = model.predict(X)
    cent = model.cluster_centers_
    return (clust_labels, cent)

clust_labels, cent = doKmeans(pm, 3)
# kmeans = pd.DataFrame(clust_labels)
# print(kmeans)
pm["kmeans"]=clust_labels

pm["kmeans"]

s

loc_b["kmeans"]=s

loc_b

s

loc_b1=loc_b

def regioncolors(counter):
    if counter['kmeans'] == 0:
        return 'green'
    elif counter['kmeans'] == 1:
        return 'blue'
    elif counter['kmeans'] == 2:
        return 'red'
    else:
        return 'darkblue'
loc_b1["color"] = loc_b1.apply(regioncolors, axis=1)
loc_b1.head()

ss=pm.groupby('kmeans').mean()

ss

sns.catplot(x='kmeans', data=pm, kind='bar');

Cluster0=np.where(pm["kmeans"]==0)
Cluster1=np.where(pm["kmeans"]==1)
Cluster2=np.where(pm["kmeans"]==2)

Cluster0

np.save("/content/drive/MyDrive/Delhi_Model_Results/Cluster0.npy", Cluster0)
np.save("/content/drive/MyDrive/Delhi_Model_Results/Cluster1.npy", Cluster1)
np.save("/content/drive/MyDrive/Delhi_Model_Results/Cluster2.npy", Cluster2)

import numpy as np

c0=np.load("/content/drive/MyDrive/Delhi_Model_Results/Cluster0.npy")
c1=np.load("/content/drive/MyDrive/Delhi_Model_Results/Cluster1.npy")
c2=np.load("/content/drive/MyDrive/Delhi_Model_Results/Cluster2.npy")

c0

asa =np.asarray(pm.mean(axis=1))

asa=pd.DataFrame(asa)
# asa.columns=loc_b['stn_name']

asa=asa.T

asa.columns=loc_b['stn_name']

asa.loc[c0]

asa[c0]

asa[c1]

asa[c2]

c0=np.squeeze(c0)
c1=np.squeeze(c1)
c2=np.squeeze(c2)

c2.shape

s=np.zeros(38)
# s[c0]==0
# s[c1]==1
# s[c2]==2
# s

a0=np.zeros(9)
a1=np.repeat(1,16)
a2=np.repeat(2,13)

aa=np.concatenate((a0,a1,a2))

aa

ss=np.concatenate((c0,c1,c2))

s[ss]=aa

c2

s

s=np.concatenate((c0,c1,c2),axis=1)
s=np.squeeze(s)
ss=np.argsort(s)

ss

loc_b1

locations=loc_b1[['lat','lon']]
locationlist = locations.values.tolist()
len(locationlist)
locationlist[7]

import folium 
from folium import plugins
from folium.plugins import HeatMap, HeatMapWithTime
# affected_area = folium.Map(location=[loc_b1.lat, loc_b1.lon], zoom_start=14,max_zoom=4,min_zoom=3,
#                           tiles='cartodbpositron',height = 500,width = '70%')
# HeatMap(data=first_month[['latitude','longitude','pm25']].groupby(['latitude','longitude']).sum().reset_index().values.tolist(),
#         radius=18, max_zoom=14).add_to(affected_area)
# affected_area

map = folium.Map(location=[28.498571000000002, 77.26484], zoom_start=12,tiles = 'Stamen Toner')
for point in range(0, 38):
    folium.Marker(locationlist[point], popup='ID:'+str(loc_b1['id'][point])+' '+loc_b1['stn_name'][point], icon=folium.Icon(color=loc_b1["color"][point], icon_color='white', icon='male', angle=0, prefix='fa')).add_to(map)
map

affected_area

!pip install pyproj

!pip install geos

conda install basemap

import mpl_toolkits
mpl_toolkits.__path__.append('/usr/lib/python2.7/dist-packages/mpl_toolkits/')
from mpl_toolkits.basemap import Basemap

import sys
from mpl_toolkits.basemap import Basemap
fig = plt.figure(figsize=(8, 8))
m = Basemap(projection='lcc', resolution=None,
            width=8E6, height=8E6, 
            lat_0=45, lon_0=-100,)
m.etopo(scale=0.5, alpha=0.5)

# Map (long, lat) to (x, y) for plotting
x, y = m(-122.3, 47.6)
plt.plot(x, y, 'ok', markersize=5)
plt.text(x, y, ' Seattle', fontsize=12);

pm=np.asarray(pm)

pmc0=np.squeeze(pm[:,c0])
pmc1=np.squeeze(pm[:,c1])
pmc2=np.squeeze(pm[:,c2])

np.mean(pmc0)

pmc0=np.ndarray.flatten(pmc0)
pmc1=np.ndarray.flatten(pmc1)
pmc2=np.ndarray.flatten(pmc2)

data=[pmc0,pmc1,pmc2]

import matplotlib.pyplot as plt
fig2, ax2 = plt.subplots()
ax2.set_title('Boxplots for 3 Clusters of Monitoring stations')
ax2.set_ylim([0,300])
ax2.boxplot(data)

m0=np.mean(pmc0)
m1=np.mean(pmc1)
m2=np.mean(pmc2)
s0=np.std(pmc0)
s1=np.std(pmc1)
s2=np.std(pmc2)

print(m0,m1,m2,s0,s1,s2)

fig2.savefig("plot.pdf")

MN=np.mean(pm.T,axis=1)

SD=np.std(pm.T,axis=1)

SD

A=pd.DataFrame(loc_b['lon'], loc_b['lat'])

A

from sklearn.cluster import DBSCAN
clustering = DBSCAN(eps=9000, min_samples=3).fit(pm.T)

clustering.labels_



from google.colab import drive
drive.mount('/content/drive')

pip install libtiff

from PIL import Image
im = Image.open('/content/drive/MyDrive/Delhi/sdat_1336_1_20211008_044623705.tif')
im.show()

from skimage import io
import matplotlib.pyplot as plt

# read the image stack
img = io.imread('/content/drive/MyDrive/Delhi/sdat_1336_1_20211008_044623705.tif')
# show the image
plt.imshow(mol,cmap='gray')