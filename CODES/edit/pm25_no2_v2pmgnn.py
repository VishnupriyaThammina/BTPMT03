
import torch
def format_pytorch_version(version):
  return version.split('+')[0]
 
TORCH_version = torch.__version__
TORCH = format_pytorch_version(TORCH_version)
 
def format_cuda_version(version):
  return 'cu' + version.replace('.', '')
 
CUDA_version = torch.version.cuda
CUDA = format_cuda_version(CUDA_version)

#@title Import Packages
import os
import sys
import torch
from collections import OrderedDict
from scipy.spatial import distance
from torch_geometric.utils import dense_to_sparse, to_dense_adj
from geopy.distance import geodesic
from metpy.units import units
import metpy.calc as mpcalc
from bresenham import bresenham
import yaml
# import util
# from util import config
import numpy as np
from datetime import datetime
import arrow
import metpy.calc as mpcalc
from metpy.units import units
from torch.utils import data
from torch.utils.data import Dataset
import os
import sys
proj_dir = os.path.dirname(os.path.abspath('/home/mainak.iiits/Dataset/multivar'))
sys.path.append(proj_dir)
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import arrow
import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import pickle
import glob
import shutil
from prettytable import PrettyTable
# from model.cells import GRUCell
from torch.nn import Sequential, Linear, Sigmoid
from torch_scatter import scatter_add#, scatter_sub  # no scatter sub in lastest PyG
from torch.nn import functional as F
from torch.nn import Parameter
from torch_geometric.nn import GATConv


proj_dir = os.path.dirname(os.path.abspath("/home/mainak.iiits/Dataset/"))
sys.path.append(proj_dir)
proj_dir1="/home/mainak.iiits/Dataset/"
from numpy import genfromtxt
rh = genfromtxt('/home/mainak.iiits/Dataset/multivar/stn_extrafeat.csv', delimiter=',')
rh=rh[1:,4:]

# from util import config
 
 
 
city_fp = os.path.join(proj_dir1, '/home/mainak.iiits/Dataset/multivar/stn_coord_j2.txt')
# altitude_fp = os.path.join(proj_dir1, 'data/altitude.npy')
 
 
class Graph():
    def __init__(self):
        # self.dist_thres = 3
        # self.alti_thres = 1200
        self.use_altitude = True
 
        # self.altitude = self._load_altitude()
        self.nodes = self._gen_nodes()
        self.node_attr = self._add_node_attr()
        self.node_num = len(self.nodes)
        # print(self.node_num)
        self.edge_index, self.edge_attr = self._gen_edges()
        # if self.use_altitude:
        #     self._update_edges()
        self.edge_num = self.edge_index.shape[1]
        self.adj = to_dense_adj(torch.LongTensor(self.edge_index))[0]
        #self._update_edges()
 
    # def _load_altitude(self):
    #     assert os.path.isfile(altitude_fp)
    #     altitude = np.load(altitude_fp)
    #     return altitude
 
    # def _lonlat2xy(self, lon, lat, is_aliti):
    #     if is_aliti:
    #         lon_l = 100.0
    #         lon_r = 128.0
    #         lat_u = 48.0
    #         lat_d = 16.0
    #         res = 0.05
    #     else:
    #         lon_l = 103.0
    #         lon_r = 122.0
    #         lat_u = 42.0
    #         lat_d = 28.0
    #         res = 0.125
    #     x = np.int64(np.round((lon - lon_l - res / 2) / res))
    #     y = np.int64(np.round((lat_u + res / 2 - lat) / res))
    #     # print(x.shape,y.shape)
    #     return x, y
 
    def _gen_nodes(self):
        nodes = OrderedDict()
        with open(city_fp,  encoding="utf16") as f:
            for line in f:
                # print(line)
                idx, stn, lon, lat = line.rstrip('\n').split(',')
                idx = int(idx)
                lon, lat = float(lon), float(lat)
                # x, y = self._lonlat2xy(lon, lat, True)
                # altitude = self.altitude[y, x]
                nodes.update({idx: {'stn': stn, 'lon': lon, 'lat': lat}})
        return nodes
 
    def _add_node_attr(self):
        node_attr = []
        altitude_arr = []
        for i in self.nodes:
          # print(i)
            altitude = rh[i,1]
            altitude_arr.append(altitude)
        altitude_arr = np.stack(altitude_arr)
        node_attr = np.stack([altitude_arr], axis=-1)
        return node_attr
 
    def traverse_graph(self):
        lons = []
        lats = []
        citys = []
        idx = []
        for i in self.nodes:
            idx.append(i)
            city = self.nodes[i]['stn']
            lon, lat = self.nodes[i]['lon'], self.nodes[i]['lat']
            lons.append(lon)
            lats.append(lat)
            citys.append(city)
        return idx, citys, lons, lats
 
    def gen_lines(self):
 
        lines = []
        for i in range(self.edge_index.shape[1]):
            src, dest = self.edge_index[0, i], self.edge_index[1, i]
            src_lat, src_lon = self.nodes[src]['lat'], self.nodes[src]['lon']
            dest_lat, dest_lon = self.nodes[dest]['lat'], self.nodes[dest]['lon']
            lines.append(([src_lon, dest_lon], [src_lat, dest_lat]))
        # print(lines)
        return lines
 
    def _gen_edges(self):
        coords = []
        lonlat = {}
        for i in self.nodes:
            coords.append([self.nodes[i]['lon'], self.nodes[i]['lat']])
        dist = distance.cdist(coords, coords, 'euclidean')
        #print(dist)
        # adj = np.zeros((self.node_num, self.node_num), dtype=np.uint8)
        # adj[dist <= self.dist_thres] = 1
        # assert adj.shape == dist.shape
        # dist = dist * adj
        edge_index, dist = dense_to_sparse(torch.tensor(dist))
        edge_index, dist = edge_index.numpy(), dist.numpy()
        # print(dist.shape)
        direc_arr = []
        dist_kilometer = []
        for i in range(edge_index.shape[1]):
            src, dest = edge_index[0, i], edge_index[1, i]
            src_lat, src_lon = self.nodes[src]['lat'], self.nodes[src]['lon']
            dest_lat, dest_lon = self.nodes[dest]['lat'], self.nodes[dest]['lon']
            src_location = (src_lat, src_lon)
            dest_location = (dest_lat, dest_lon)
            dist_km = geodesic(src_location, dest_location).kilometers
            v, u = src_lat - dest_lat, src_lon - dest_lon
 
            u = u * units.meter / units.second
            v = v * units.meter / units.second
            direc = mpcalc.wind_direction(u, v)._magnitude
 
            direc_arr.append(direc)
            dist_kilometer.append(dist_km)
 
        direc_arr = np.stack(direc_arr)
        dist_arr = np.stack(dist_kilometer)
        #print(dist_arr)
        attr = np.stack([dist_arr, direc_arr], axis=-1)
        # print(attr.shape)
        return edge_index, attr
 
    def _update_edges(self):
        edge_index = []
        edge_attr = []
        for i in range(self.edge_index.shape[1]):
            src, dest = self.edge_index[0, i], self.edge_index[1, i]
            src_y, src_x = self.nodes[src]['lat'], self.nodes[src]['lon']
            dest_y, dest_x = self.nodes[dest]['lat'], self.nodes[dest]['lon']
            # print()
            # src_x, src_y = self.src_lon, src_lat, True)
            # dest_x, dest_y = self._lonlat2xy(dest_lon, dest_lat, True)
            # points = np.asarray(list(bresenham(src_y, src_x, dest_y, dest_x))).transpose((1,0))
            # altitude_points = self.altitude[points[0], points[1]]
            altitude_src = rh[sr,1]
            altitude_dest = rh[dest,1]
            # if np.sum(altitude_points - altitude_src > self.alti_thres) < 3 and \
            #    np.sum(altitude_points - altitude_dest > self.alti_thres) < 3:
                # edge_index.append(self.edge_index[:,i])
                # edge_attr.append(self.edge_attr[i])
        
        self.edge_index = np.stack(edge_index, axis=1)
        self.edge_attr = np.stack(edge_attr, axis=0)
        print(edge_attr)
 
 
if __name__ == '__main__':
  graph = Graph()

totaldata=np.load("/home/mainak.iiits/Dataset/multivar/3_hr_data_june2023.npy",allow_pickle=True)
# wswd=np.load("/content/drive/MyDrive/Delhi Lockdown/GNNDelhi/DelhiData_3hr_sample_wswd.npy")
totaldata=np.float32(totaldata)
pm10=totaldata[:,:,-1]
totaldata=np.delete(totaldata, [1,2,3,8,14,15],2)



print(totaldata.shape)

#0	   1 	2	   3	4  	5 	6	  7 	8	     9 	10	11	12       13	  14
#pm25	pm10	SO2	nh3	rh	ws	wd	at	prec	v10	u10	pbl	kindex	press	temp

class HazeData(data.Dataset):
    def __init__(self, graph,
                 hist_len=16,
                 pred_len=8,
                 dataset_num=1,
                 flag = 'Train'):
 
        if flag == 'Train':
            start_time_str = [[2020, 1, 1], 'GMT']
            end_time_str = [[2022, 5, 26], 'GMT']
#         elif flag == 'Val':
#             start_time_str = [[2021, 7, 16], 'GMT']
#             end_time_str = [[2021, 9, 30], 'GMT']
        elif flag == 'Test':
            start_time_str = [[2022, 5,26], 'GMT']
            end_time_str = [[2022,12,31], 'GMT']
        else:
            raise Exception('Wrong Flag!')
        self.start_time = self._get_time(start_time_str)
        self.end_time = self._get_time(end_time_str)
        self.data_start = self._get_time([[2020, 1, 1, 0, 0], 'GMT'])
        #config['dataset']['data_start']
        #config['dataset']['data_end']
        self.data_end = self._get_time([[2022, 12, 31, 23, 30], 'GMT'])
        # file_dir = config['filepath']
        # self.knowair_fp = file_dir
        # print(self.knowair_fp)
        self.graph = graph
        self._load_npy()
        self._gen_time_arr()
        self._process_time()
       # self._process_feature2()
        self.feature = np.float32(self.feature)
        self.pm25 = np.float32(self.pm25)
        self.pm10= np.float32(self.pm10)

        seq_len = hist_len + pred_len
        # print(seq_len)
        self._add_time_dim(seq_len)
        self._calc_mean_std()
        # self._norm()
        # self._get_time()
 
    def _norm(self):
        # print(np.isnan(self.feature))
        self.feature = (self.feature - self.feature_mean) / self.feature_std
        # print(np.isnan(self.feature))
        self.pm25 = (self.pm25 - self.pm25_mean) / self.pm25_std
        self.pm10 = (self.pm10 - self.pm10_mean) / self.pm10_std
        # print(self.pm25.shape)
 
 
    def _add_time_dim(self, seq_len):
 
        def _add_t(arr, seq_len):
            t_len = arr.shape[0]
            assert t_len > seq_len
            arr_ts = []
            for i in range(seq_len, t_len):
                arr_t = arr[i-seq_len:i]
                arr_ts.append(arr_t)
            arr_ts = np.stack(arr_ts, axis=0)
            return arr_ts
        
        self.pm25 = _add_t(self.pm25, seq_len)
        self.pm10 = _add_t(self.pm10, seq_len)
        # print(self.pm25.shape)
        self.feature = _add_t(self.feature, seq_len)
        self.time_arr = _add_t(self.time_arr, seq_len)
        # print("time se panga",self.time_arr.shape)
 
    def _calc_mean_std(self):
        self.featmean = self.feature.mean(axis=(0,1,2))
        # print("featmean",self.featmean.shape)
        self.featstd = self.feature.std(axis=(0,1,2))
        # print("featstd",self.featstd)
        self.wind_mean = self.featmean[3:5]
        self.wind_std = self.featstd[3:5]
        self.pm25_mean = self.pm25.mean()
        self.pm25_std = self.pm25.std()
        self.pm10_mean = self.pm10.mean()
        self.pm10_std = self.pm10.std()
        self.feature_mean = self.featmean
        self.feature_std = self.featstd
        # self.wind_mean = self.featmean[-2:]
        # self.wind_std = self.featstd[-2:]
        # self.pm25_mean = PM_mean
        # self.pm25_std = PM_std
 
 #     def _process_feature1(self):
#         # metero_var = config['data']['metero_var']
#         # # print(metero_var)
#         # metero_use = config['experiments']['metero_use']
#         # metero_idx = [metero_var.index(var) for var in metero_use]
#         # # print(metero_idx)
#         # self.feature = self.feature[:,:,metero_idx]
#         print(self.feature.shape)
#         # u = self.feature[:, :, -2] * units.meter / units.second
#         # v = self.feature[:, :, -1] * units.meter / units.second
#         # speed = 3.6 * mpcalc.wind_speed(u, v)._magnitude
#         # direc = mpcalc.wind_direction(u, v)._magnitude
#         # print(speed.shape)
#         # speed =self.feature[:,:,2]
#         # direc=self.feature[:,:,1]
#         h_arr = []
#         # print("fygj",h_arr)
#         w_arr = []
#         count1=0
#         for i in self.time_arrow:
#             # print("fygj",i)
#             # count1=count1+1
#             # print(count1)
#             h_arr.append(i.hour)
#             w_arr.append(i.isoweekday())
#         h_arr = np.stack(h_arr, axis=-1)
#         w_arr = np.stack(w_arr, axis=-1)
#         h_cos=np.cos(2 * np.pi * h_arr/23.0)
#         h_sin=np.sin(2 * np.pi * h_arr/23.0)
#         w_cos=np.cos(2 * np.pi * w_arr/6.0)
#         w_sin=np.sin(2 * np.pi * w_arr/6.0)
#         # print("Age h_arr",w_arr.shape)
#         h_cos = np.repeat(h_cos[:, None], self.graph.node_num, axis=1)
#         w_cos = np.repeat(w_cos[:, None], self.graph.node_num, axis=1)
#         h_sin = np.repeat(h_sin[:, None], self.graph.node_num, axis=1)
#         w_sin = np.repeat(w_sin[:, None], self.graph.node_num, axis=1)
#         # print("Pore h_arr",h_arr.shape)
#         # print("warr",w_arr.shape)
#         # print("featdim",self.feature.shape)
#         self.feature = np.concatenate([self.feature, h_cos[:, :, None], w_cos[:, :, None], h_sin[:, :, None], w_sin[:, :, None]], axis=-1)
#         # print("feature_clubbed",self.feature.shape)
#     def _process_feature2(self):   
#         h_arr = []
#         w_arr = []
#         count1=0
#         for i in self.time_arrow:
#             h_arr.append(i.hour)
#             w_arr.append(i.isoweekday())     
#         h_arr = np.stack(h_arr, axis=-1)
#         w_arr = np.stack(w_arr, axis=-1)
#         h_arr = np.repeat(h_arr[:, None], self.graph.node_num, axis=1)
#         w_arr = np.repeat(w_arr[:, None], self.graph.node_num, axis=1)
#         self.feature = np.concatenate([self.feature,h_arr[:,:,None],w_arr[:,:,None]],axis=-1)
    def _process_time(self):
        start_idx = self._get_idx(self.start_time)
        # print( "start",start_idx)
        end_idx = self._get_idx(self.end_time)
        # print("end",end_idx)
        self.pm25 = self.pm25[start_idx: end_idx+1, :]
        self.pm10 = self.pm10[start_idx: end_idx+1, :]
        self.feature = self.feature[start_idx: end_idx+1, :]
        # print("feat_today",self.feature.shape)
        # print(np.isnan(self.feature))
        self.time_arr = self.time_arr[start_idx: end_idx+1]
        self.time_arrow = self.time_arrow[start_idx: end_idx + 1]
        # print(time_arrow.shape)
 
    def _gen_time_arr(self):
        self.time_arrow = []
        self.time_arr = []
        count=0
        for time_arrow in arrow.Arrow.interval('hours', self.data_start, self.data_end.shift(hours=+1), 1):
            # print(time_arrow.int_timestamp)
            # print(1)
            # count +=1
            # print("time_arr", count)
            self.time_arrow.append(time_arrow[0])
            self.time_arr.append(time_arrow[0].int_timestamp)
        self.time_arr = np.stack(self.time_arr, axis=-1)
        # print("time_arrow",self.time_arrow)
        # print("load",self.time_arr.shape)
    def _load_npy(self):
        # print("load",self.knowair_fp)
        # self.knowair = Hour_3_sample
        self.feature = totaldata[:,:,2:]
        # print("feattter",self.feature.shape)
        self.pm25 = totaldata[:,:,0]
        self.pm10 = pm10
        # print("self", self.pm25.type)
    def _get_idx(self, t):
        t0 = self.data_start
        return int((t.timestamp() - t0.timestamp()) / (60*60))
 
    def _get_time(self, time_yaml):
        arrow_time = arrow.get(datetime(*time_yaml[0]), time_yaml[1])
        return arrow_time
    def __len__(self):
        return len(self.pm25) 
 
    def __getitem__(self, index):
        return self.pm25[index],self.pm10[index],self.feature[index], self.time_arr[index]
        # torch.tensor(list(self.pm25[index]),dtype=torch.float32) , torch.tensor(list(self.feature[index]),dtype=torch.float32), torch.tensor(list(self.time_arr[index]),dtype=torch.float32)
    
 
if __name__ == '__main__':
    # from graph import Graph
  #  graph = Graph()
    train_data = HazeData( graph,flag='Train')
#     val_data = HazeData( flag='Val')
    test_data = HazeData(graph, flag='Test')
    
    print(test_data.feature.shape)
    print(len(train_data))
    print(len(test_data))


# val_data = HazeData(graph, flag='Val')
#test_data = HazeData(graph, flag='Test')

#@title Data_Normalization
train_data.feature=(train_data.feature-train_data.feature_mean)/train_data.feature_std
train_data.pm25=(train_data.pm25-train_data.pm25_mean)/train_data.pm25_std
test_data.feature=(test_data.feature-train_data.feature_mean)/train_data.feature_std
test_data.pm25=(test_data.pm25-train_data.pm25_mean)/train_data.pm25_std
# val_data.feature=(val_data.feature-train_data.feature_mean)/train_data.feature_std
# val_data.pm25=(val_data.pm25-train_data.pm25_mean)/train_data.pm25_std

train_data.pm10=(train_data.pm10-train_data.pm10_mean)/train_data.pm10_std
test_data.pm10=(test_data.pm10-test_data.pm10_mean)/test_data.pm10_std





#@title Graph_GNN_module
Graph_GNN = "" #@param {type:"string"}

class GraphGNN(nn.Module):
    def __init__(self, device, edge_index, edge_attr, in_dim, out_dim, wind_mean, wind_std):
        super(GraphGNN, self).__init__()
        self.device = device
        self.edge_index = torch.LongTensor(edge_index).to(self.device)
        self.edge_attr = torch.Tensor(np.float32(edge_attr))
        self.edge_attr_norm = (self.edge_attr - self.edge_attr.mean(dim=0)) / self.edge_attr.std(dim=0)
        self.w = Parameter(torch.rand([1]))
        self.b = Parameter(torch.rand([1]))
        self.wind_mean = torch.from_numpy(np.asarray(wind_mean)).to(self.device)
        self.wind_std = torch.from_numpy(np.asarray(wind_std)).to(self.device)
        # print("achi_kaka")
        e_h = 27
        e_out = 27
        n_out = 2
        self.edge_mlp = Sequential(Linear(in_dim * 2 + 2 + 1, e_h),
                                   Sigmoid(),
                                   Linear(e_h, e_out),
                                   Sigmoid(),
                                   )
        self.node_mlp = Sequential(Linear(e_out, n_out),
                                   Sigmoid(),
                                   )

    def forward(self, x):
        self.edge_index = self.edge_index.to(self.device)
        self.edge_attr = self.edge_attr.to(self.device)
        self.w = self.w.to(self.device)
        self.b = self.b.to(self.device)
        # print("x",x.shape)
        edge_src, edge_target = self.edge_index
        # print("src",edge_src.shape)
        node_src = x[:, edge_src]
        node_target = x[:, edge_target]
        # print("nodesrc",node_src.shape)
        # print(self.wind_std[None,None,:].shape)
        src_wind = node_src[:,:,-3:-1] * self.wind_std[None,None,:] + self.wind_mean[None,None,:]
        # print("srcwnd",src_wind.shape)
        src_wind_speed = src_wind[:, :, 0]
        src_wind_direc = src_wind[:,:,1]
        # print("srcwndsped",src_wind_speed)
        # print("edge_attr",self.edge_attr.shape)
        self.edge_attr_ = self.edge_attr[None, :, :].repeat(node_src.size(0), 1, 1)
        city_dist = self.edge_attr_[:,:,0]
        city_direc = self.edge_attr_[:,:,1]
        # print("citydist",city_dist.shape)
        theta = torch.abs(city_direc - src_wind_direc)
        edge_weight = F.relu(3 * src_wind_speed * torch.cos(theta) / city_dist)
        edge_weight = edge_weight.to(self.device)
        # print("ew",edge_weight.shape)
        edge_attr_norm = self.edge_attr_norm[None, :, :].repeat(node_src.size(0), 1, 1).to(self.device)
        # print("ean",edge_attr_norm.shape)
        # print("node_src",node_src.shape)
        out = torch.cat([node_src, node_target, edge_attr_norm, edge_weight[:,:,None]], dim=-1)
        # print("out1",out.shape)
        out = self.edge_mlp(out)
        # print("out2",out.shape)
        out_add = scatter_add(out, edge_target, dim=1, dim_size=x.size(1))
        # out_sub = scatter_sub(out, edge_src, dim=1, dim_size=x.size(1))
        out_sub = scatter_add(out.neg(), edge_src, dim=1, dim_size=x.size(1))  # For higher version of PyG.
        # print("outadd",out_add.shape)
        # print("outsub",out_sub.shape)
        out = out_add + out_sub
        out = self.node_mlp(out)

        return out

#@title GRU-Ce
class GRUCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        x = x.view(-1, x.size(-1))
        #print("aa",x.size(-1))
        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)
        # print("grugateh",gate_h.shape)
        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)
        
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + (resetgate * h_n))

        hy = newgate + inputgate * (hidden - newgate)

        return hy

#@title LSTMcell

class LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        hx, cx = hidden
        x = x.view(-1, x.size(-1))
        gates = self.x2h(x) + self.h2h(hx)
        gates = gates.squeeze()
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)
        cy = torch.mul(cx, forgetgate) + torch.mul(ingate, cellgate)
        hy = torch.mul(outgate, F.tanh(cy))
        return (hy, cy)




##Multivar LSTM
import torch
from torch import nn
#from model.cells import LSTMCell


class LSTM(nn.Module):
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device):
        super(LSTM, self).__init__()
        self.device = device
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.city_num = city_num
        self.batch_size = batch_size
        self.in_dim = in_dim
        self.hid_dim = 32
        self.out_dim = 2
        self.fc_in = nn.Linear(self.in_dim, self.hid_dim)
        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)#(hidden_dim, 2)(1 for o3, 1 for nox)
        self.lstm_cell = LSTMCell(self.hid_dim, self.hid_dim)

    def forward(self, pm25_hist,pm10_hist,feature):
        pm25_pred = []
        pm10_pred = []
        h0 = torch.zeros(self.batch_size * self.city_num, self.hid_dim).to(self.device)
        hn = h0
        c0 = torch.zeros(self.batch_size * self.city_num, self.hid_dim).to(self.device)
        cn = c0
        # print('pm25_hist',pm25_hist.shape)
        # print('pm10_hist',pm10_hist.shape)
        # print('feat',feature.shape)
        for i in range(self.hist_len):
            x = torch.cat((torch.unsqueeze(pm25_hist[:,i],-1),torch.unsqueeze(pm10_hist[:,i],-1), feature[:, i]), dim=-1)
            # print(x.shape)
            x = self.fc_in(x)
            hn, cn = self.lstm_cell(x, (hn, cn))
            xn = hn.view(self.batch_size, self.city_num, self.hid_dim)
            xn = self.fc_out(xn)  
        #     pm25_pred.append(xn[:,:,0])
        #     pm10_pred.append(xn[:,:,1])
        # pm25_pred = torch.stack(pm25_pred, dim=-1)
        # pm10_pred = torch.stack(pm10_pred, dim=-1)   
        for i in range(self.pred_len):
            x = torch.cat((xn, feature[:, self.hist_len+i]), dim=-1)
#             print('hello')
#             print('hello  ',x.shape)
            x = self.fc_in(x)
            hn, cn = self.lstm_cell(x, (hn, cn))
            xn = hn.view(self.batch_size, self.city_num, self.hid_dim)
            xn = self.fc_out(xn)  
            pm25_pred.append(xn[:,:,0])
            pm10_pred.append(xn[:,:,1])
        pm25_pred = torch.stack(pm25_pred, dim=-1)
        pm10_pred = torch.stack(pm10_pred, dim=-1)
        pm25_pred=pm25_pred.permute(0,2,1)
        pm10_pred=pm10_pred.permute(0,2,1)
        return  pm25_pred, pm10_pred

import torch.nn.functional as F
from torch_geometric.nn import ChebConv
import torch
from torch import nn


class GC_LSTM(nn.Module):
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device, edge_index):
        super(GC_LSTM, self).__init__()
        self.edge_index = torch.LongTensor(edge_index)
        self.edge_index = self.edge_index.view(2, 1, -1).repeat(1, batch_size, 1) + torch.arange(batch_size).view(1, -1, 1) * city_num
        self.edge_index = self.edge_index.view(2, -1)
        self.device = device
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.city_num = city_num
        self.batch_size = batch_size
        self.in_dim = in_dim
        self.hid_dim = 32
        self.out_dim = 2
        self.gcn_out = 2
        self.conv = ChebConv(self.in_dim, self.gcn_out, K=2)
        self.lstm_cell = LSTMCell(self.in_dim + self.gcn_out, self.hid_dim)
        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)

    def forward(self, pm25_hist,pm10_hist, feature):
        self.edge_index = self.edge_index.to(self.device)
        pm25_pred = []
        pm10_pred = []
        h0 = torch.zeros(self.batch_size * self.city_num, self.hid_dim).to(self.device)
        hn = h0
        c0 = torch.zeros(self.batch_size * self.city_num, self.hid_dim).to(self.device)
        cn = c0
        # print("histpm",pm25_hist.shape)
        # print("feature",feature.shape)
        for i in range(self.hist_len):
            #x = torch.cat((torch.unsqueeze(pm25_hist[:,i],-1), feature[:, i]), dim=-1)
            x = torch.cat((torch.unsqueeze(pm25_hist[:,i],-1),torch.unsqueeze(pm10_hist[:,i],-1), feature[:, i]), dim=-1)
            x_gcn = x.contiguous()
            x_gcn = x_gcn.view(self.batch_size * self.city_num, -1)
            x_gcn = F.sigmoid(self.conv(x_gcn, self.edge_index))
            x_gcn = x_gcn.view(self.batch_size, self.city_num, -1)
            x = torch.cat((x, x_gcn), dim=-1)
            hn, cn = self.lstm_cell(x, (hn, cn))
            xn = hn.view(self.batch_size, self.city_num, self.hid_dim)
            xn = self.fc_out(xn)
            # print("xn",xn.shape)
            # pm25_pred.append(xn)
        for i in range(self.pred_len):
            x = torch.cat((xn, feature[:, self.hist_len+i]), dim=-1)
            x_gcn = x.contiguous()
            x_gcn = x_gcn.view(self.batch_size * self.city_num, -1)
            x_gcn = F.sigmoid(self.conv(x_gcn, self.edge_index))
            x_gcn = x_gcn.view(self.batch_size, self.city_num, -1)
            x = torch.cat((x, x_gcn), dim=-1)
            hn, cn = self.lstm_cell(x, (hn, cn))
            xn = hn.view(self.batch_size, self.city_num, self.hid_dim)
            xn = self.fc_out(xn)
            pm25_pred.append(xn[:,:,0])
            pm10_pred.append(xn[:,:,1])
        pm25_pred = torch.stack(pm25_pred, dim=-1)
        pm10_pred = torch.stack(pm10_pred, dim=-1)
        pm25_pred=pm25_pred.permute(0,2,1)
        pm10_pred=pm10_pred.permute(0,2,1)
        return pm25_pred, pm10_pred


class PM25_GNN(nn.Module):
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device, edge_index, edge_attr, wind_mean, wind_std):
        super(PM25_GNN, self).__init__()

        self.device = device
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.city_num = city_num
        self.batch_size = batch_size

        self.in_dim = in_dim
        self.hid_dim = 64
        self.out_dim = 2
        self.gnn_out = 2
        self.in_dim=6
        self.fc_in = nn.Linear(self.in_dim, self.hid_dim)
        self.graph_gnn = GraphGNN(self.device, edge_index, edge_attr, self.in_dim, self.gnn_out, wind_mean, wind_std)
        self.gru_cell = GRUCell(self.in_dim + self.gnn_out, self.hid_dim)
        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)

    def forward(self, pm25_hist, pm10_hist, feature):
        pm25_pred = []
        pm10_pred = []
        h0 = torch.zeros(self.batch_size * self.city_num, self.hid_dim).to(self.device)
        hn = h0
        # xn = pm25_hist[:, -1]

        for i in range(self.hist_len):
          #x = torch.cat((torch.unsqueeze(pm25_hist[:,i],-1), feature[:, i]), dim=-1)
          x = torch.cat((torch.unsqueeze(pm25_hist[:,i],-1),torch.unsqueeze(pm10_hist[:,i],-1), feature[:, i]), dim=-1)
          xn_gnn = x
          xn_gnn = xn_gnn.contiguous()
          xn_gnn = self.graph_gnn(xn_gnn)
          # print("xgnn",xn_gnn.shape)
         
          x = torch.cat([xn_gnn, x], dim=-1)
          # print("x",x.shape)
          hn = self.gru_cell(x, hn)
          xn = hn.view(self.batch_size, self.city_num, self.hid_dim)
          xn = self.fc_out(xn)
        for i in range(self.pred_len):
            x = torch.cat((xn, feature[:, self.hist_len + i]), dim=-1)

            xn_gnn = x
            xn_gnn = xn_gnn.contiguous()
            xn_gnn = self.graph_gnn(xn_gnn)
            x = torch.cat([xn_gnn, x], dim=-1)
         
            hn = self.gru_cell(x, hn)
            xn = hn.view(self.batch_size, self.city_num, self.hid_dim)
            xn = self.fc_out(xn)
            pm25_pred.append(xn[:,:,0])
            pm10_pred.append(xn[:,:,1])
        pm25_pred = torch.stack(pm25_pred, dim=-1)
        pm10_pred = torch.stack(pm10_pred, dim=-1)
        pm25_pred=pm25_pred.permute(0,2,1)
        pm10_pred=pm10_pred.permute(0,2,1)
        return pm25_pred, pm10_pred
#test_loss, predict_epoch_pm25, predict_epoch_pm10,label_epoch_pm25,label_epoch_pm10, time_epoch = test(test_loader, model)

#@title Training_module
import logging
lr = 0.001 #@param {type:"number"}
in_dim_temp_gru = 32 #@param {type:"number"}
in_dim_gnn_gru = 25 #@param {type:"number"}
weight_decay = 0.0001 #@param {type:"number"}
exp_repeat = 1 #@param {type:"number"}
epochs = 50 #@param {type:"number"}
in_dim = 6 #@param {type:"number"}
#save_numpy = True #@param {type:"raw"}
def get_logger(file_path):

        logger = logging.getLogger('gal')
        log_format = '%(asctime)s | %(message)s'
        formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        logger.setLevel(logging.INFO)
        return logger
torch.set_num_threads(1)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
 
# graph = Graph()
city_num = graph.node_num
# s=np.concatenate((Cluster0,Cluster1,Cluster2),axis=1)
# s=np.squeeze(s)
# ss=np.argsort(s)
batch_size = 32
#epochs = 50

gradient_accumulations = 4
scaler = GradScaler()
hist_len =16
pred_len = 8
#weight_decay = 0.0001
early_stop = 5
results_dir = "/home/mainak.iiits/Dataset/multivar/results"
dataset_num = 1
# exp_model = 'PM25_GNN'
# model= PM25_GNN
#exp_repeat = 5
save_npy = True
criterion = nn.MSELoss()
 
# print(config['dataset'][3]['train_start'])
# print('**************************')
# train_data = HazeData(graph, hist_len, pred_len, dataset_num, flag='Train')
# # val_data = HazeData(graph, hist_len, pred_len, dataset_num, flag='Val')
# test_data = HazeData(graph, hist_len, pred_len, dataset_num, flag='Test')
#in_dim = 14
wind_mean, wind_std = train_data.wind_mean, train_data.wind_std
pm25_mean, pm25_std = train_data.pm25_mean, train_data.pm25_std
pm10_mean, pm10_std = train_data.pm10_mean, train_data.pm10_std

# print("wndmn",wind_mean.shape)
 
def get_metric(predict_epoch, label_epoch):
    haze_threshold = 60
    predict_epoch=predict_epoch[:,hist_len:]
    label_epoch=label_epoch[:,hist_len:]
    # print("pred",predict_epoch.shape)
    # print("lab",label_epoch.shape)
    predict_haze = predict_epoch >= haze_threshold
    predict_clear = predict_epoch < haze_threshold
    label_haze = label_epoch >= haze_threshold
    label_clear = label_epoch < haze_threshold
    hit = np.sum(np.logical_and(predict_haze, label_haze))
    miss = np.sum(np.logical_and(label_haze, predict_clear))
    falsealarm = np.sum(np.logical_and(predict_haze, label_clear))
    csi = hit / (hit + falsealarm + miss)
    pod = hit / (hit + miss)
    far = falsealarm / (hit + falsealarm)
    # predict = predict_epoch[:,:,:,0].transpose((0,2,1))
    # label = label_epoch[:,:,:,0].transpose((0,2,1))
    # predict = predict.reshape((-1, predict.shape[-1]))
    # label = label.reshape((-1, label.shape[-1]))
    mae = np.mean(np.mean(np.abs(predict_epoch - label_epoch), axis=1))
    rmse = np.mean(np.sqrt(np.mean(np.square(predict_epoch - label_epoch), axis=1)))
    predict_epoch=np.ndarray.flatten(predict_epoch)
    label_epoch=np.ndarray.flatten(label_epoch)
    max=np.ndarray.max(label_epoch)
    min=np.ndarray.min(label_epoch)
    nrmse=rmse/(max-min)
    # print("dsff",predict_epoch.shape)
    # mape = np.mean(np.abs((predict_epoch - label_epoch)/label_epoch), axis=1)
    r2=r2_score(label_epoch,predict_epoch)
    return rmse, nrmse, mae,r2, csi, pod, far

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
# def get_exp_info():
#     exp_info =  '============== Train Info ==============\n' + \
#                 'Dataset number: %s\n' % dataset_num + \
#                 'Train: %s --> %s\n' % (train_data.start_time, train_data.end_time) + \
#                 'Val: %s --> %s\n' % (val_data.start_time, val_data.end_time) + \
#                 'Test: %s --> %s\n' % (test_data.start_time, test_data.end_time) + \
#                 'City number: %s\n' % city_num + \
#                 'Use metero: %s\n' % config['experiments']['metero_use'] + \
#                 'batch_size: %s\n' % batch_size + \
#                 'epochs: %s\n' % epochs + \
#                 'hist_len: %s\n' % hist_len + \
#                 'pred_len: %s\n' % pred_len + \
#                 'weight_decay: %s\n' % weight_decay + \
#                 'early_stop: %s\n' % early_stop + \
#                 'lr: %s\n' % lr + \
#                 '========================================\n'
#     return exp_info
 
 
# def get_model():
#     if exp_model == 'MLP':
#         return MLP(hist_len, pred_len, in_dim)
#     elif exp_model == 'LSTM':
#         return LSTM(hist_len, pred_len, in_dim, city_num, batch_size, device)
#     elif exp_model == 'GRU':
#         return GRU(hist_len, pred_len, in_dim, city_num, batch_size, device)
#     elif exp_model == 'nodesFC_GRU':
#         return nodesFC_GRU(hist_len, pred_len, in_dim, city_num, batch_size, device)
#     elif exp_model == 'GC_LSTM':
#         return GC_LSTM(hist_len, pred_len, in_dim, city_num, batch_size, device, graph.edge_index)
#     elif exp_model == 'PM25_GNN':
#         return PM25_GNN(hist_len, pred_len, in_dim, city_num, batch_size, device, graph.edge_index, graph.edge_attr, wind_mean, wind_std)
#     elif exp_model == 'PM25_GNN_nosub':
#         return PM25_GNN_nosub(hist_len, pred_len, in_dim, city_num, batch_size, device, graph.edge_index, graph.edge_attr, wind_mean, wind_std)
#     else:
#         raise Exception('Wrong model name!')
 
 
def train(train_loader, model, optimizer):
    model.train()
    train_loss = 0
    for batch_idx, data in tqdm(enumerate(train_loader)):
        pm25,pm10, feature, time_arr = data
        # print("training",pm25.shape)
        #print("featingdtype",feature.shape)
        pm25 = pm25.to(device)
        pm10=pm10.to(device)
        feature = feature.to(device)
        # print("feay",feature.shape)
        pm25_label = pm25[:, hist_len:]
        pm25_hist = pm25[:, :hist_len]
        pm10_label = pm10[:, hist_len:]
        pm10_hist = pm10[:, :hist_len]
        with autocast():
             pm25_pred,pm10_pred = model(pm25_hist,pm10_hist, feature)
            #  print("pred",pm25_pred.shape)
            #  print("label",pm25_label.shape)
             loss1= criterion(torch.squeeze(pm25_pred), pm25_label)
             loss2= criterion(torch.squeeze(pm10_pred), pm10_label)
            #  loss2 = criterion(pred_gru, pm25_label) 
             loss=loss1+loss2
        # l2_lambda = 0.01
        # l2_reg = torch.tensor(0.).to(device)
        # for param in model.parameters():
        #     l2_reg += torch.norm(param)
        # loss += l2_lambda * l2_reg
        L1_reg = torch.tensor(0., requires_grad=True)
        for name, param in model.named_parameters():
           if 'weight' in name:
              L1_reg = L1_reg + torch.norm(param, 1)

        loss = loss + 10e-4 * L1_reg     
        scaler.scale(loss / gradient_accumulations).backward()
        optimizer.step()
        if (batch_idx + 1) % gradient_accumulations == 0:
              scaler.step(optimizer)
              scaler.update()
              optimizer.zero_grad()
        train_loss += loss.item()
    train_loss /= batch_idx + 1
    return train_loss
 
 
def val(val_loader, model):
    model.eval()
    val_loss = 0
    for batch_idx, data in tqdm(enumerate(val_loader)):
        pm25, feature, time_arr = data
        pm25 = pm25.to(device)
        feature = feature.to(device)
        pm25_label = pm25[:, hist_len:]
        pm25_hist = pm25[:, :hist_len]
        pm25_pred = model(pm25_hist, feature)
        loss1 = criterion(pm25_pred, pm25_label)
        loss2 = criterion(pred_gru, pm25_label) 
        loss=loss1+loss2
        val_loss += loss.item()
 
    val_loss /= batch_idx + 1
    return val_loss
 
 
def test(test_loader, model):
    model.eval()
    predict_list_pm25 = []
    label_list_pm25 = []
    predict_list_pm10=[]
    label_list_pm10=[]
    time_list = []
    test_loss = 0
    for batch_idx, data in enumerate(test_loader):
        pm25,pm10, feature, time_arr = data
        # print("training",pm25.shape)
        #print("featingdtype",feature.shape)
        pm25 = pm25.to(device)
        pm10=pm10.to(device)
        feature = feature.to(device)
        pm25_label = pm25[:, hist_len:]
        pm25_hist = pm25[:, :hist_len]
        pm10_label = pm10[:, hist_len:]
        pm10_hist = pm10[:, :hist_len]
        pm25_pred,pm10_pred = model(pm25_hist,pm10_hist, feature)
        loss1 = criterion(torch.squeeze(pm25_pred), pm25_label)
        loss2 = criterion(torch.squeeze(pm10_pred), pm10_label)

        # loss2 = criterion(pred_gru, pm25_label) 
        loss=loss1+loss2
        test_loss += loss.item()
        # print("pm25_hist",pm25_hist.shape)
        # print("pm25_pred",pm25_pred.shape)
        pm25_pred_val = np.concatenate([pm25_hist.cpu().detach().numpy(), pm25_pred.cpu().detach().numpy()], axis=1) * pm25_std + pm25_mean
        pm25_label_val = pm25.cpu().detach().numpy() * pm25_std + pm25_mean
        pm10_pred_val = np.concatenate([pm10_hist.cpu().detach().numpy(), pm10_pred.cpu().detach().numpy()], axis=1) * pm10_std + pm10_mean
        pm10_label_val = pm25.cpu().detach().numpy() * pm10_std + pm10_mean
        predict_list_pm25.append(np.expand_dims(pm25_pred_val,-1))
        label_list_pm25.append(np.expand_dims(pm25_label_val,-1))
        predict_list_pm10.append(np.expand_dims(pm10_pred_val,-1))
        label_list_pm10.append(np.expand_dims(pm10_label_val,-1))
        time_list.append(time_arr.cpu().detach().numpy())
 
    test_loss /= batch_idx + 1
 
    predict_epoch_pm25 = np.concatenate(predict_list_pm25, axis=0)
    label_epoch_pm25 = np.concatenate(label_list_pm25, axis=0)
    predict_epoch_pm10 = np.concatenate(predict_list_pm10, axis=0)
    label_epoch_pm10 = np.concatenate(label_list_pm10, axis=0)
    time_epoch = np.concatenate(time_list, axis=0)
    predict_epoch_pm25[predict_epoch_pm25 < 0] = 0
    predict_epoch_pm10[predict_epoch_pm10 < 0] = 0
    return test_loss, predict_epoch_pm25, predict_epoch_pm10,label_epoch_pm25,label_epoch_pm10, time_epoch

 
def get_mean_std(data_list):
    data = np.asarray(data_list)
    return data.mean(), data.std()
 
 
def main():
    # exp_info = get_exp_info()
    # print(exp_info)
 
    exp_time = arrow.now().format('YYYYMMDDHHmmss')
 
    train_loss_list, val_loss_list, test_loss_list, rmse_list, mae_list, csi_list, pod_list, far_list = [], [], [], [], [], [], [], []
 
    for exp_idx in range(exp_repeat):
        print('\nNo.%2d experiment ~~~' % exp_idx)
 
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        #val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)
        model=PM25_GNN(hist_len, pred_len, in_dim, city_num, batch_size, device, graph.edge_index, graph.edge_attr, wind_mean, wind_std)

        #model=ModelNew1(hist_len, pred_len, in_dim,in_dim_gnn_gru, city_num,Cluster0,Cluster1,Cluster2,ss, batch_size, device, graph.edge_index, graph.edge_attr, wind_mean, wind_std, in_dim_temp_gru)
        # model.load_state_dict(torch.load('/content/drive/MyDrive/Delhi_Model_Results/16_8/1/PM25GNN/20220103040118/00/13/model.pth'))
        # model.eval()
        model=model.to(device)
        print(str(model))
        count_parameters(model)
        #optimizer = torch.optim.Adam(model.parameters(), lr=0.0015)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
 
        #exp_model_dir = os.path.join(results_dir, '%s_%s' % (hist_len, pred_len), str(dataset_num), 'PM25GNN', str(exp_time), '%02d' % exp_idx)
        exp_model_dir = os.path.join(results_dir,'Multivar', '%s_%s' % (hist_len, pred_len),'PM25-PM10', 'LSTM', str(exp_time), '%02d' % exp_idx)

        # if not os.path.exists(exp_model_dir):
        #     os.makedirs(exp_model_dir)
        # model_fp = os.path.join(exp_model_dir, 'model.pth')
        logger=get_logger(results_dir+'%02d'% exp_idx +'logger_LSTM.log')
 
        val_loss_min = 100000
        best_epoch = 0
 
        # train_loss_, val_loss_ = 0, 0
        train_loss_=0
        epoch_rmse_pm25=[]
        epoch_mae_pm25=[]
        epoch_nrmse_pm25=[]
        epoch_r2_pm25=[]
        epoch_csi_pm25=[]
        epoch_pod_pm25=[]
        epoch_far_pm25=[]
        epoch_rmse_pm10=[]
        epoch_mae_pm10=[]
        epoch_nrmse_pm10=[]
        epoch_r2_pm10=[]
        epoch_csi_pm10=[]
        epoch_pod_pm10=[]
        epoch_far_pm10=[]
        TrainLoss=[]
        TestLoss=[]
 
        for epoch in range(epochs):
            print('\nTrain epoch %s:' % (epoch))
 
            train_loss = train(train_loader, model, optimizer)
            # val_loss = val(val_loader, model)
            print('train',train)
            print('train_loss: %.4f' % train_loss)
            # print('val_loss: %.4f' % val_loss)
            test_loss, predict_epoch_pm25, predict_epoch_pm10,label_epoch_pm25,label_epoch_pm10, time_epoch = test(test_loader,model)
                # print(test_loss,'#######################################')
            # train_loss_, val_loss_ = train_loss, val_loss
            train_loss_=  train_loss
            rmse_pm25, nrmse_pm25,mae_pm25,r2_pm25,csi_pm25, pod_pm25, far_pm25 = get_metric(predict_epoch_pm25, label_epoch_pm25)
            rmse_pm10,nrmse_pm10, mae_pm10,r2_pm10,csi_pm10, pod_pm10, far_pm10 = get_metric(predict_epoch_pm10, label_epoch_pm10)
            epoch_rmse_pm25.append(rmse_pm25)
            epoch_mae_pm25.append(mae_pm25)
            
            epoch_rmse_pm10.append(rmse_pm10)
            epoch_mae_pm10.append(mae_pm10)
            

            epoch_nrmse_pm25.append(nrmse_pm25)
            epoch_r2_pm25.append(r2_pm25)
            epoch_csi_pm25.append(csi_pm25)
            epoch_pod_pm25.append(pod_pm25)
            epoch_far_pm25.append(far_pm25)
            epoch_nrmse_pm10.append(nrmse_pm10)
            epoch_r2_pm10.append(r2_pm10)
            epoch_csi_pm10.append(csi_pm10)
            epoch_pod_pm10.append(pod_pm10)
            epoch_far_pm10.append(far_pm10)
            TrainLoss.append(train_loss)
            TestLoss.append(test_loss)
            # if epoch - best_epoch > early_stop:
            #     break
 
            # if rmse < val_loss_min:
            # val_loss_min = rmse
            best_epoch = epoch
            exp_model_dir1= os.path.join(exp_model_dir,str(epoch))
            if not os.path.exists(exp_model_dir1):
                os.makedirs(exp_model_dir1)
            model_fp = os.path.join(exp_model_dir1, 'model.pth')
            # print('Minimum val loss!!!')
            torch.save(model.state_dict(), model_fp)
                # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('Save model: %s' % model_fp)

            print('Train loss: %0.4f, Test loss: %0.4f, RMSE_pm25: %0.2f, NRMSE_pm25: %0.2f, MAE_pm25: %0.2f,R2_pm25: %0.4f, CSI_pm25: %0.4f, POD_pm25: %0.4f, FAR_pm25: %0.4f' % (train_loss_, test_loss, rmse_pm25,nrmse_pm25, mae_pm25,r2_pm25, csi_pm25, pod_pm25, far_pm25))
            print('RMSE_nox: %0.2f, NRMSE_nox: %0.2f,MAE_nox: %0.2f,R2_nox: %0.4f, CSI_nox: %0.4f, POD_nox: %0.4f, FAR_nox: %0.4f' % (rmse_pm10, nrmse_pm10,mae_pm10,r2_pm10, csi_pm10, pod_pm10, far_pm10))
 
            # print('Train loss: %0.4f, Val loss: %0.4f, Test loss: %0.4f, RMSE: %0.2f, MAE: %0.2f,R2: %0.4f, CSI: %0.4f, POD: %0.4f, FAR: %0.4f' % (train_loss_, val_loss_, test_loss, rmse, mae,r2, csi, pod, far))
            # print('Train loss: %0.4f, Test loss: %0.4f, RMSE: %0.2f,NRMSE: %0.4f, MAE: %0.2f,R2: %0.4f, CSI: %0.4f, POD: %0.4f, FAR: %0.4f' % (train_loss_, test_loss, rmse,nrmse, mae,r2, csi, pod, far))
            # logger.info('Epoch: {}/{:.3f}---Train,{:.3f}----Test,RMSE{:.3f},NRMSE{:.4f},MAE{:.3f},R2{:.3f},CSI{:.3f},POD{:.3f},FAR{:.3f}\n'.format(epoch,train_loss_,test_loss,rmse,nrmse,mae,r2,csi,pod,far))

            if save_npy:
                np.save(os.path.join(exp_model_dir1, 'predict_pm25.npy'), predict_epoch_pm25)
                np.save(os.path.join(exp_model_dir1, 'label_pm25.npy'), label_epoch_pm25)
                np.save(os.path.join(exp_model_dir1, 'time.npy'), time_epoch)
                np.save(os.path.join(exp_model_dir1, 'predict_pm10.npy'), predict_epoch_pm10)
                np.save(os.path.join(exp_model_dir1, 'label_pm10.npy'), label_epoch_pm10)
                # np.save(os.path.join(exp_model_dir1, 'time.npy'), time_epoch)
        #if save_npy:
        np.save(os.path.join(exp_model_dir, 'epoch_rmse.npy'), np.array(epoch_rmse_pm25))
        np.save(os.path.join(exp_model_dir, 'epoch_mae.npy'), np.array(epoch_mae_pm25))
        np.save(os.path.join(exp_model_dir, 'epoch_nrmse.npy'), np.array(epoch_nrmse_pm25))
        np.save(os.path.join(exp_model_dir, 'epoch_r2.npy'), np.array(epoch_r2_pm25))
        np.save(os.path.join(exp_model_dir, 'epoch_csi.npy'), np.array(epoch_csi_pm25))
        np.save(os.path.join(exp_model_dir, 'epoch_pod.npy'), np.array(epoch_pod_pm25))
        np.save(os.path.join(exp_model_dir, 'epoch_far.npy'), np.array(epoch_far_pm25))
        np.save(os.path.join(exp_model_dir, 'epoch_trainloss.npy'), np.array(TrainLoss))
        np.save(os.path.join(exp_model_dir, 'epoch_testloss.npy'), np.array(TestLoss))
        np.save(os.path.join(exp_model_dir, 'epoch_rmse.npy'), np.array(epoch_rmse_pm10))
        np.save(os.path.join(exp_model_dir, 'epoch_mae.npy'), np.array(epoch_mae_pm10))
        np.save(os.path.join(exp_model_dir, 'epoch_nrmse.npy'), np.array(epoch_nrmse_pm10))
        np.save(os.path.join(exp_model_dir, 'epoch_r2.npy'), np.array(epoch_r2_pm10))
        np.save(os.path.join(exp_model_dir, 'epoch_csi.npy'), np.array(epoch_csi_pm10))
        np.save(os.path.join(exp_model_dir, 'epoch_pod.npy'), np.array(epoch_pod_pm10))
        np.save(os.path.join(exp_model_dir, 'epoch_far.npy'), np.array(epoch_far_pm10))
        # np.save(os.path.join(exp_model_dir, 'epoch_trainloss.npy'), np.array(TrainLoss))
        # np.save(os.path.join(exp_model_dir, 'epoch_testloss.npy'), np.array(TestLoss))
      #test_loss=val_loss_
        train_loss_list.append(train_loss_)
        # val_loss_list.append(val_loss_)
        test_loss_list.append(test_loss)
        # rmse_list.append(rmse)
        # mae_list.append(mae)
        # csi_list.append(csi)
        # pod_list.append(pod)
        # far_list.append(far)
        #if save_npy: 
        # np.save(os.path.join(exp_model_dir, 'trainlosslist.npy'), np.array(train_loss_list))
        # np.save(os.path.join(exp_model_dir, 'test_loss_list.npy'), np.array(test_loss_list))
        # np.save(os.path.join(exp_model_dir, 'rmse_list.npy'), np.array(rmse_list))
        # np.save(os.path.join(exp_model_dir, 'mae_list.npy'), np.array(mae_list))
        # np.save(os.path.join(exp_model_dir, 'csi_list.npy'), np.array(csi_list))
        # np.save(os.path.join(exp_model_dir, 'pod_list.npy'), np.array(pod_list))
        # np.save(os.path.join(exp_model_dir, 'far_list.npy'), np.array(far_list))
        # plot1=plt.figure(1)
        # plt.plot(epoch_rmse, label='RMSE')
        # plt.xlabel('Epochs')
        # plt.ylabel('RMSE(in μg/m3)')
        # file='RMSE.pdf'
        # plt.savefig(os.path.join(exp_model_dir1,file))
        # plt.clf()
        # plot2=plt.figure(2)
        # plt.plot(epoch_mae, label='MAE')
        # plt.xlabel('Epochs')
        # plt.ylabel('MAE(in μg/m3)')
        # file='MAE.pdf'
        # plt.savefig(os.path.join(exp_model_dir1,file))
        # plt.clf()
        # plot3=plt.figure(3)
        # plt.plot(epoch_rmse, label='RMSE')
        # plt.plot(epoch_mae, label='MAE')
        # plt.xlabel('Epochs')
        # plt.ylabel('(in μg/m3)')
        # file='RMSE-MAE.pdf'
        # plt.savefig(os.path.join(exp_model_dir1,file))
        # plt.clf()
        # plot4=plt.figure(4)
        # plt.plot(TrainLoss, label='Train_loss')
        # plt.plot(TestLoss, label='Test_loss')
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # file='Train_TestLoss.pdf'
        # plt.savefig(os.path.join(exp_model_dir1,file))
        # plt.clf()
        # print('\nNo.%2d experiment results:' % exp_idx)
        # print(
        #     'Train loss: %0.4f, Test loss: %0.4f, RMSE: %0.2f, NRMSE:%0.2f ,MAE: %0.2f, R2: %0.2f, CSI: %0.4f, POD: %0.4f, FAR: %0.4f' % (
        #     train_loss_, test_loss, rmse,nrmse, mae,r2, csi, pod, far))
 
    exp_metric_str = '---------------------------------------\n' + \
                     'learningRate... %0.4f' % (lr) +\
                     'weight_decay %0.4f' % (weight_decay)
 
    metric_fp = os.path.join(os.path.dirname(exp_model_dir), 'metric.txt')
    with open(metric_fp, 'w') as f:
        f.write(str(model))
        f.write(exp_metric_str)
 
    print('=========================\n')
    print(exp_metric_str)
    print(str(model))
    print(metric_fp)
 
 
if __name__ == '__main__':
    main()

