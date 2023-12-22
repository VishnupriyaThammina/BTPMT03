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
proj_dir = os.path.dirname(os.path.abspath('/content/drive/MyDrive/BTP-03'))
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

from google.colab import drive
drive.mount('/content/drive')

#@title Graph_Structure
Graph_Structure = "Graph_Struct" #@param {type:"string"}
#Graph


proj_dir = os.path.dirname(os.path.abspath("/content/drive/MyDrive/BTP-03"))
sys.path.append(proj_dir)
proj_dir1="/content/drive/MyDrive/BTP-03"
from numpy import genfromtxt
dem = genfromtxt('/content/drive/MyDrive/BTP-03/DATASET/dem.csv', delimiter=',')
lcss = genfromtxt('/content/drive/MyDrive/BTP-03/DATASET/lcss.csv', delimiter=',')


# from util import config



city_fp = os.path.join(proj_dir1, '/content/drive/MyDrive/BTP-03/DATASET/st_lulcIndia.csv')
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
        self.edge_index, self.edge_attr = self._gen_edges()
        self.edge_num = self.edge_index.shape[1]
        self.adj = to_dense_adj(torch.LongTensor(self.edge_index))[0]


    def _gen_nodes(self):
        nodes = OrderedDict()
        with open(city_fp, 'r') as f:
            for line in f:
                idx, stn, city, lon, lat = line.rstrip('\n').split(',')
                idx = int(idx)
                lon, lat = float(lon), float(lat)
                nodes.update({idx: {'stn': stn, 'city':city, 'lon': lon, 'lat': lat}})
        return nodes

    def _add_node_attr(self):
        node_attr = []
        altitude_arr = []
        lcss_arr = []
        for i in self.nodes:
            altitude = dem[i]
            lcssvar = lcss[i]
            altitude_arr.append(altitude)
            lcss_arr.append(lcssvar)
        altitude_arr = np.stack(altitude_arr)
        lcss_arr = np.stack(lcss_arr)

        node_attr = np.stack([lcss_arr,altitude_arr], axis=-1)
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

        return edge_index, attr



if __name__ == '__main__':
  graph = Graph()

#@title Dataset_load
# import numpy as np

Dataset = "" #@param {type:"string"}
totaldata=np.load("/content/drive/MyDrive/BTP-03/DATASET/Dataset.npy", allow_pickle = True)
# wswd=np.load("/content/drive/MyDrive/Delhi Lockdown/GNNDelhi/DelhiData_3hr_sample_wswd.npy")
totaldata=np.float32(totaldata)

totaldata = np.delete(totaldata, 2, axis=2) # axis=2 indicates the column index for the 'aod' variable

print(totaldata.shape)

u = totaldata[:, :, -2] * units.meter / units.second
v = totaldata[:, :, -1] * units.meter / units.second
speed = mpcalc.wind_speed(u, v)._magnitude
direc = mpcalc.wind_direction(u, v)._magnitude

totaldata[:,:, -2] = speed
totaldata[:,:,-1] = direc

#np.save("/content/drive/MyDrive/BTP-03/DATASET/Dataset_With_WSWD.npy", totaldata)

totaldata[:,:,-2]

totaldata[:,:,-1]

"""**ORDER OF VARIABLES:**
0.blh 1.aod 2.kx 3.pm25 4.sp 5.t2m 6.tp 7.u10 8.v10

NEW ORDER: 7. speed, 8. direction
"""

#@title Dataset_object
class HazeData(data.Dataset):
    def __init__(self, graph,
                       hist_len=16,
                       pred_len=8,
                       dataset_num=1,
                       flag='Train'):

        if flag == 'Train':
            start_time_str = [[2021, 1, 1,3,0], 'GMT']
            end_time_str = [[2022, 5, 26,23,0], 'GMT']
        elif flag == 'Val':
            start_time_str = [[2022, 6, 2], 'GMT']
            end_time_str = [[2022, 8, 6], 'GMT']
        elif flag == 'Test':
            start_time_str = [[2022, 5 ,27, 0, 0], 'GMT']
            end_time_str = [[2022,12,31, 23, 0], 'GMT']
        else:
            raise Exception('Wrong Flag!')
        self.start_time = self._get_time(start_time_str)
        self.end_time = self._get_time(end_time_str)
        self.data_start = self._get_time([[2021, 1, 1, 3, 0], 'GMT'])

        self.data_end = self._get_time([[2022, 12, 31, 23, 0], 'GMT'])
        # file_dir = config['filepath']
        # self.knowair_fp = file_dir
        # print(self.knowair_fp)
        self.graph = graph
        self._load_npy()
        self._gen_time_arr()
        self._process_time()
        self._process_feature2()
        self.feature = np.float32(self.feature)
        self.pm25 = np.float32(self.pm25)


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
        # print(self.pm25.shape)


    def _add_time_dim(self, seq_len):

        def _add_t(arr, seq_len):
            t_len = arr.shape[0]
            #print(t_len)
            #print(seq_len)
            assert t_len > seq_len
            arr_ts = []
            for i in range(seq_len, t_len):
                arr_t = arr[i-seq_len:i]
                arr_ts.append(arr_t)
            arr_ts = np.stack(arr_ts, axis=0)
            return arr_ts

        self.pm25 = _add_t(self.pm25, seq_len)
        # print(self.pm25.shape)
        self.feature = _add_t(self.feature, seq_len)
        self.time_arr = _add_t(self.time_arr, seq_len)
        # print("time se panga",self.time_arr.shape)

    def _calc_mean_std(self):
        self.featmean = self.feature.mean(axis=(0,1,2))
        # print("featmean",self.featmean.shape)
        self.featstd = self.feature.std(axis=(0,1,2))
        # print("featstd",self.featstd)


        self.pm25_mean = self.pm25.mean()
        self.pm25_std = self.pm25.std()
        self.feature_mean = self.featmean
        self.feature_std = self.featstd
        self.wind_mean = self.featmean[-4:-2]
        self.wind_std = self.featstd[-4:-2]
        # self.pm25_mean = PM_mean
        # self.pm25_std = PM_std

    def _process_feature1(self):

        u = self.feature[:, :, -2] * units.meter / units.second
        v = self.feature[:, :, -1] * units.meter / units.second
        speed = mpcalc.wind_speed(u, v)._magnitude
        direc = mpcalc.wind_direction(u, v)._magnitude

        h_arr = []
        w_arr = []
        for i in self.time_arrow:
            h_arr.append(i.hour)
            w_arr.append(i.isoweekday())
        h_arr = np.stack(h_arr, axis=-1)
        w_arr = np.stack(w_arr, axis=-1)
        h_arr = np.repeat(h_arr[:, None], self.graph.node_num, axis=1)
        w_arr = np.repeat(w_arr[:, None], self.graph.node_num, axis=1)

        self.feature = np.concatenate([self.feature, h_arr[:, :, None], w_arr[:, :, None],
                                       speed[:, :, None], direc[:, :, None]
                                       ], axis=-1)


    def _process_feature2(self):
        h_arr = []
        w_arr = []
        count1=0
        for i in self.time_arrow:
            h_arr.append(i.hour)
            w_arr.append(i.isoweekday())
        h_arr = np.stack(h_arr, axis=-1)
        w_arr = np.stack(w_arr, axis=-1)
        h_arr = np.repeat(h_arr[:, None], self.graph.node_num, axis=1)
        w_arr = np.repeat(w_arr[:, None], self.graph.node_num, axis=1)
        self.feature = np.concatenate([self.feature,h_arr[:,:,None],w_arr[:,:,None]],axis=-1)
    def _process_time(self):
        start_idx = self._get_idx(self.start_time)
        # print( "start",start_idx)
        end_idx = self._get_idx(self.end_time)
        # print("end",end_idx)
        self.pm25 = self.pm25[start_idx: end_idx+1, :]
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
        for time_arrow in arrow.Arrow.interval('hours', self.data_start, self.data_end.shift(hours=+3), 3):
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
        # self.knowair = totaldata
        self.feature = np.delete(totaldata,2,2)
        # print("feattter",self.feature.shape)
        self.pm25 = totaldata[:,:,2]

        # print("self", self.pm25.type)
    def _get_idx(self, t):
        t0 = self.data_start
        return int((t.timestamp() - t0.timestamp()) / (60*60*3))

    def _get_time(self, time_yaml):
        arrow_time = arrow.get(datetime(*time_yaml[0]), time_yaml[1])
        return arrow_time
    def __len__(self):
        return len(self.pm25)

    def __getitem__(self, index):
        return self.pm25[index], self.feature[index], self.time_arr[index]
        # torch.tensor(list(self.pm25[index]),dtype=torch.float32) , torch.tensor(list(self.feature[index]),dtype=torch.float32), torch.tensor(list(self.time_arr[index]),dtype=torch.float32)


if __name__ == '__main__':
    # from graph import Graph
    # graph = Graph()
    train_data = HazeData(graph,flag='Train')
    #val_data = HazeData(flag='Val')
    test_data = HazeData(graph,flag='Test')

train_data.wind_mean

totaldata[:,:,-1].mean(axis=(0,1))

#@title Data_Normalization
train_data.feature=(train_data.feature-train_data.feature_mean)/train_data.feature_std
train_data.pm25=(train_data.pm25-train_data.pm25_mean)/train_data.pm25_std
test_data.feature=(test_data.feature-train_data.feature_mean)/train_data.feature_std
test_data.pm25=(test_data.pm25-train_data.pm25_mean)/train_data.pm25_std

train_data.feature[-1]

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

#@title GRU-Cell

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
        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)

        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + (resetgate * h_n))

        hy = newgate + inputgate * (hidden - newgate)

        return hy

#@title LSTM

class LSTM(nn.Module):
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device):
        super(LSTM, self).__init__()
        self.device = device
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.city_num = city_num
        self.batch_size = batch_size
        self.in_dim = in_dim
        self.hid_dim =64
        self.m = nn.Dropout(p=0.2)
        self.out_dim = 1
        self.fc_in = nn.Linear(self.in_dim, self.hid_dim)
        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)
        self.lstm_cell = LSTMCell(self.hid_dim, self.hid_dim)

    def forward(self, pm25_hist, feature):
        pm25_pred = []
        h0 = torch.zeros(self.batch_size * self.city_num, self.hid_dim).to(self.device)
        hn = h0
        c0 = torch.zeros(self.batch_size * self.city_num, self.hid_dim).to(self.device)
        cn = c0

        for i in range(self.hist_len):

            x = torch.cat((torch.unsqueeze(pm25_hist[:,i],-1), feature[:, i]), dim=-1)

            x = self.fc_in(x)
            hn, cn = self.lstm_cell(x, (hn, cn))
            xn = hn.view(self.batch_size, self.city_num, self.hid_dim)
            xn = self.fc_out(xn)


        for i in range(self.pred_len):

            x = torch.cat((xn, feature[:, self.hist_len+i]), dim=-1)

            x = self.fc_in(x)
            hn, cn = self.lstm_cell(x, (hn, cn))
            xn = hn.view(self.batch_size, self.city_num, self.hid_dim)
            xn = self.fc_out(xn)
            pm25_pred.append(xn)
        pm25_pred = torch.stack(pm25_pred, dim=1)
        pm25_pred = torch.squeeze(pm25_pred)
        return pm25_pred

#@title GRU

class GRU(nn.Module):
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device):
        super(GRU, self).__init__()
        self.device = device
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.city_num = city_num
        self.batch_size = batch_size
        self.in_dim = in_dim
        self.hid_dim = 32
        self.out_dim = 1
        self.fc_in = nn.Linear(self.in_dim, self.hid_dim)
        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)
        self.gru_cell = GRUCell(self.hid_dim, self.hid_dim)

    def forward(self, pm25_hist, feature):
        pm25_pred = []
        h0 = torch.zeros(self.batch_size * self.city_num, self.hid_dim).to(self.device)
        hn = h0
        #xn = pm25_hist[:, -1]
        for i in range(self.hist_len):
            # print(torch.unsqueeze(xn,-1).shape)
            # print(feature[:, self.hist_len+i].shape)
            x = torch.cat((torch.unsqueeze(pm25_hist[:,i],-1), feature[:, i]), dim=-1)
            #print("GRU x shape:",x.shape)
            x = self.fc_in(x)
            hn = self.gru_cell(x, hn)
            xn = hn.view(self.batch_size, self.city_num, self.hid_dim)
            xn = self.fc_out(xn)

        for i in range(self.pred_len):
            x = torch.cat((xn, feature[:, self.hist_len+i]), dim=-1)
            x = self.fc_in(x)
            hn = self.gru_cell(x, hn)
            xn = hn.view(self.batch_size, self.city_num, self.hid_dim)
            xn = self.fc_out(xn)
            pm25_pred.append(xn)
        pm25_pred = torch.stack(pm25_pred, dim=1)
        pm25_pred = torch.squeeze(pm25_pred)
        return pm25_pred

#@title GC_LSTM

from torch_geometric.nn import ChebConv
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
        self.out_dim = 1
        self.gcn_out = 2
        self.conv = ChebConv(self.in_dim, self.gcn_out, K=2)
        self.lstm_cell = LSTMCell(self.in_dim + self.gcn_out, self.hid_dim)
        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)

    def forward(self, pm25_hist, feature):
        self.edge_index = self.edge_index.to(self.device)
        pm25_pred = []
        h0 = torch.zeros(self.batch_size * self.city_num, self.hid_dim).to(self.device)
        hn = h0
        c0 = torch.zeros(self.batch_size * self.city_num, self.hid_dim).to(self.device)
        cn = c0
        # print("histpm",pm25_hist.shape)
        # print("feature",feature.shape)
        for i in range(self.hist_len):
            x = torch.cat((torch.unsqueeze(pm25_hist[:,i],-1), feature[:, i]), dim=-1)
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
            pm25_pred.append(xn)

        pm25_pred = torch.squeeze(torch.stack(pm25_pred, dim=1))
        # print("pred",pm25_pred.shape)#[32, 8, 40, 1])
        return pm25_pred

from torch.nn import Parameter


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
        e_h = 48
        e_out = 48
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
        #print("x",x.shape)
        #print("edge",self.edge_index.shape)
        edge_src, edge_target = self.edge_index
        #print("EDGE_SRC", edge_src.shape)
        #print("EDGE TARGET", edge_target.shape)
        #print("node_src",x[:,:].shape)
        #print("WIND STD SHAPE:",self.wind_std[None,None,:].shape)
        #print("WIND MEAN SHAPE:", wind_mean.shape)

        node_src = x[:, edge_src]
        node_target = x[:, edge_target]
        #print("nodesrc",node_src.shape)

        src_wind = node_src[:,:,-4:-2] * self.wind_std[None,None,:] + self.wind_mean[None,None,:]
        #print("srcwnd",src_wind.shape)
        src_wind_speed = src_wind[:, :, 0]
        src_wind_direc = src_wind[:,:,1]
        # print("srcwndsped",src_wind_speed)
        #print("edge_attr",self.edge_attr.shape)
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

        #print("OUT SHAPE", out.shape)
        out = self.edge_mlp(out)
        #print("out2",out.shape)
        out_add = scatter_add(out, edge_target, dim=1, dim_size=x.size(1))
        # out_sub = scatter_sub(out, edge_src, dim=1, dim_size=x.size(1))
        out_sub = scatter_add(out.neg(), edge_src, dim=1, dim_size=x.size(1))  # For higher version of PyG.
        # print("outadd",out_add.shape)
        # print("outsub",out_sub.shape)
        out = out_add + out_sub
        out = self.node_mlp(out)

        return out

from torch_geometric.nn import knn_graph

#@title PM2.5-GNN

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
        self.out_dim = 1
        self.gnn_out = 2

        self.fc_in = nn.Linear(self.in_dim, self.hid_dim)
        self.graph_gnn = GraphGNN(self.device, edge_index, edge_attr, self.in_dim, self.gnn_out, wind_mean, wind_std)
        #edge-conv
        self.gru_cell = GRUCell(self.in_dim + self.gnn_out, self.hid_dim)
        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)

    def forward(self, pm25_hist, feature):
        pm25_pred = []
        h0 = torch.zeros(self.batch_size*self.city_num, self.hid_dim).to(self.device)
        hn = h0
        pm25_hist = pm25_hist.unsqueeze(-1)
        xn = pm25_hist[:, -1]
        for i in range(self.pred_len):

            #print("TENSOR 1 (x):", xn.unsqueeze(-1).shape)
            #print("Tensor 2:", feature[:, self.hist_len + i].shape)
            x = torch.cat((xn, feature[:, self.hist_len + i]), dim=-1)

            xn_gnn = x
            xn_gnn = xn_gnn.contiguous()
            xn_gnn = xn_gnn.view(self.batch_size, self.city_num, -1)
            xn_gnn = self.graph_gnn(xn_gnn)
            xn_gnn = xn_gnn.view(self.batch_size, self.city_num, -1)

            #edgeconv(x)
            x = torch.cat([xn_gnn, x], dim=-1)

            #print("x shape:", x.shape)
            #print("hn:", hn.shape)
            hn = self.gru_cell(x, hn)
            xn = hn.view(self.batch_size, self.city_num, self.hid_dim)
            xn = self.fc_out(xn)
            pm25_pred.append(xn)

        pm25_pred = torch.stack(pm25_pred, dim=1)

        return pm25_pred

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch

#@title Training_module
import logging
lr = 0.0015 #@param {type:"number"}
#in_dim_lstm = 9
#in_dim_temp_gru = 32 #@param {type:"number"}
#in_dim_gnn_gru = 25 #@param {type:"number"}
#weight_decay = 0.0001 #@param {type:"number"}
exp_repeat = 1 #@param {type:"number"}
#epochs = 50 #@param {type:"number"}
in_dim = 10 #@param {type:"number"}
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
epochs = 50

gradient_accumulations = 4
scaler = GradScaler()
hist_len = 16
pred_len = 8
weight_decay = 0.0001
early_stop = 5
lr = 0.0015
#in_dim_temp_gru=32
#in_dim_gnn_gru=25
results_dir = "/content/drive/MyDrive/BTP-03/PM25-GNN/Results"
dataset_num = 1
# exp_model = 'PM25_GNN'
# model= PM25_GNN
exp_repeat = 1
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
#print("wndmn",wind_mean)
#print("WIND STD:", wind_std)

def get_metric(predict_epoch, label_epoch):
    haze_threshold = 60
    predict_epoch=predict_epoch[:,hist_len:]
    label_epoch=label_epoch[:,hist_len:]
    print("pred",predict_epoch.shape)
    print("lab",label_epoch.shape)
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
    print("pred",predict_epoch.shape)
    print("lab",label_epoch.shape)
    print("pred",predict_epoch)
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


def train(train_loader, model, optimizer):
    model.train()
    train_loss = 0
    for batch_idx, data in tqdm(enumerate(train_loader)):
        pm25, feature, time_arr = data
        # print("training",pm25.shape)
        #print("featingdtype",feature.shape)
        pm25 = pm25.to(device)

        #print("PMSHAPE",pm25.shape)
        feature = feature.to(device)
        pm25_label = pm25[:, hist_len:]
        pm25_hist = pm25[:, :hist_len]

        #print("label",pm25_label.shape)
        #print("Feature",feature.shape)
        # print(pm25_label)
        with autocast():
             pm25_pred = model(pm25_hist, feature)


             pm25_pred = torch.squeeze(pm25_pred)
             #print("pred",pm25_pred.shape)
             loss= criterion(pm25_pred, pm25_label)
            #  loss2 = criterion(pred_gru, pm25_label)
            #  loss=loss1+loss2
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




def test(test_loader, model):
    model.eval()
    predict_list = []
    label_list = []
    time_list = []
    test_loss = 0
    for batch_idx, data in enumerate(test_loader):
        pm25, feature, time_arr = data
        pm25 = pm25.to(device)
        feature = feature.to(device)
        pm25_label = pm25[:, hist_len:]
        pm25_hist = pm25[:, :hist_len]
        pm25_pred = model(pm25_hist, feature)
        pm25_pred = torch.squeeze(pm25_pred)
        loss = criterion(pm25_pred, pm25_label)
        # loss2 = criterion(pred_gru, pm25_label)
        # loss=loss1+loss2
        test_loss += loss.item()

        pm25_pred_val = np.concatenate([pm25_hist.cpu().detach().numpy(), pm25_pred.cpu().detach().numpy()], axis=1) * pm25_std + pm25_mean
        pm25_label_val = pm25.cpu().detach().numpy() * pm25_std + pm25_mean
        predict_list.append(pm25_pred_val)
        label_list.append(pm25_label_val)
        time_list.append(time_arr.cpu().detach().numpy())

    test_loss /= batch_idx + 1

    predict_epoch = np.concatenate(predict_list, axis=0)
    label_epoch = np.concatenate(label_list, axis=0)
    time_epoch = np.concatenate(time_list, axis=0)
    predict_epoch[predict_epoch < 0] = 0

    return test_loss, predict_epoch, label_epoch, time_epoch


def get_mean_std(data_list):
    data = np.asarray(data_list)
    return data.mean(), data.std()



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



def main():
    # exp_info = get_exp_info()
    # print(exp_info)

    exp_time = arrow.now().format('YYYYMMDDHHmmss')

    train_loss_list, val_loss_list, test_loss_list, rmse_list, mae_list, csi_list, pod_list, far_list,r2_list = [], [], [], [], [], [], [], [], []

    for exp_idx in range(exp_repeat):
        print('\nNo.%2d experiment ~~~' % exp_idx)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        #val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=True)
        print(train_loader)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)

        model = PM25_GNN(hist_len, pred_len, in_dim, city_num, batch_size, device, graph.edge_index, graph.edge_attr, wind_mean, wind_std)
        # model.load_state_dict(torch.load('/content/drive/MyDrive/Delhi_Model_Results/16_8/1/PM25GNN/20220103040118/00/13/model.pth'))
        # model.eval()
        model=model.to(device)
        print(str(model))
        count_parameters(model)
        #optimizer = torch.optim.Adam(model.parameters(), lr=0.0015)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

        #exp_model_dir = os.path.join(results_dir, '%s_%s' % (hist_len, pred_len), str(dataset_num), 'PM25GNN', str(exp_time), '%02d' % exp_idx)
        exp_model_dir = os.path.join(results_dir, '%s_%s' % (hist_len, pred_len), 'PM2.5-GNN', str(exp_time), '%02d' % exp_idx)

        if not os.path.exists(exp_model_dir):
            os.makedirs(exp_model_dir)
        model_fp = os.path.join(exp_model_dir, 'model.pth')
        logger=get_logger(results_dir+'%02d'% exp_idx +'logger_SAGNN.log')

        val_loss_min = 100000
        best_epoch = 0

        # train_loss_, val_loss_ = 0, 0
        train_loss_=0
        epoch_rmse=[]
        epoch_mae=[]
        epoch_nrmse=[]
        epoch_r2=[]
        epoch_csi=[]
        epoch_pod=[]
        epoch_far=[]
        TrainLoss=[]
        TestLoss=[]

        for epoch in range(epochs):
            print('\nTrain epoch %s:' % (epoch))

            train_loss = train(train_loader, model, optimizer)
            # val_loss = val(val_loader, model)

            print('train_loss: %.4f' % train_loss)
            # print('val_loss: %.4f' % val_loss)

            test_loss, predict_epoch, label_epoch, time_epoch = test(test_loader, model)
                # print(test_loss,'#######################################')
            # train_loss_, val_loss_ = train_loss, val_loss
            print(predict_epoch)
            print(label_epoch)
            train_loss_=  train_loss
            rmse, nrmse,mae,r2,csi, pod, far = get_metric(predict_epoch, label_epoch)

            print("rmse",rmse)
            epoch_rmse.append(rmse)
            epoch_mae.append(mae)
            epoch_nrmse.append(nrmse)
            epoch_r2.append(r2)
            epoch_csi.append(csi)
            epoch_pod.append(pod)
            epoch_far.append(far)
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

            # print('Train loss: %0.4f, Val loss: %0.4f, Test loss: %0.4f, RMSE: %0.2f, MAE: %0.2f,R2: %0.4f, CSI: %0.4f, POD: %0.4f, FAR: %0.4f' % (train_loss_, val_loss_, test_loss, rmse, mae,r2, csi, pod, far))
            print('Train loss: %0.4f, Test loss: %0.4f, RMSE: %0.2f,NRMSE: %0.4f, MAE: %0.2f,R2: %0.4f, CSI: %0.4f, POD: %0.4f, FAR: %0.4f' % (train_loss_, test_loss, rmse,nrmse, mae,r2, csi, pod, far))
            logger.info('Epoch: {}/{:.3f}---Train,{:.3f}----Test,RMSE{:.3f},NRMSE{:.4f},MAE{:.3f},R2{:.3f},CSI{:.3f},POD{:.3f},FAR{:.3f}\n'.format(epoch,train_loss_,test_loss,rmse,nrmse,mae,r2,csi,pod,far))

            if save_npy:
                np.save(os.path.join(exp_model_dir1, 'predict.npy'), predict_epoch)
                np.save(os.path.join(exp_model_dir1, 'label.npy'), label_epoch)
                np.save(os.path.join(exp_model_dir1, 'time.npy'), time_epoch)
        #if save_npy:
        np.save(os.path.join(exp_model_dir, 'epoch_rmse.npy'), np.array(epoch_rmse))
        np.save(os.path.join(exp_model_dir, 'epoch_mae.npy'), np.array(epoch_mae))
        np.save(os.path.join(exp_model_dir, 'epoch_nrmse.npy'), np.array(epoch_nrmse))
        np.save(os.path.join(exp_model_dir, 'epoch_r2.npy'), np.array(epoch_r2))
        np.save(os.path.join(exp_model_dir, 'epoch_csi.npy'), np.array(epoch_csi))
        np.save(os.path.join(exp_model_dir, 'epoch_pod.npy'), np.array(epoch_pod))
        np.save(os.path.join(exp_model_dir, 'epoch_far.npy'), np.array(epoch_far))
        np.save(os.path.join(exp_model_dir, 'epoch_trainloss.npy'), np.array(TrainLoss))
        np.save(os.path.join(exp_model_dir, 'epoch_testloss.npy'), np.array(TestLoss))
      #test_loss=val_loss_
        train_loss_list.append(train_loss_)
        # val_loss_list.append(val_loss_)
        test_loss_list.append(test_loss)
        rmse_list.append(rmse)
        mae_list.append(mae)
        csi_list.append(csi)
        pod_list.append(pod)
        far_list.append(far)
        r2_list.append(r2)

        #if save_npy:
        np.save(os.path.join(exp_model_dir, 'trainlosslist.npy'), np.array(train_loss_list))
        np.save(os.path.join(exp_model_dir, 'test_loss_list.npy'), np.array(test_loss_list))
        np.save(os.path.join(exp_model_dir, 'rmse_list.npy'), np.array(rmse_list))
        np.save(os.path.join(exp_model_dir, 'mae_list.npy'), np.array(mae_list))
        np.save(os.path.join(exp_model_dir, 'csi_list.npy'), np.array(csi_list))
        np.save(os.path.join(exp_model_dir, 'pod_list.npy'), np.array(pod_list))
        np.save(os.path.join(exp_model_dir, 'far_list.npy'), np.array(far_list))
        np.save(os.path.join(exp_model_dir, 'r2.npy'), np.array(r2_list))

        print('\nNo.%2d experiment results:' % exp_idx)
        print(
            'Train loss: %0.4f, Test loss: %0.4f, RMSE: %0.2f, NRMSE:%0.2f ,MAE: %0.2f, R2: %0.2f, CSI: %0.4f, POD: %0.4f, FAR: %0.4f' % (
            train_loss_, test_loss, rmse,nrmse, mae,r2, csi, pod, far))

    exp_metric_str = '---------------------------------------\n' + \
                     'train_loss | mean: %0.4f std: %0.4f\n' % (get_mean_std(train_loss_list)) + \
                     'test_loss  | mean: %0.4f std: %0.4f\n' % (get_mean_std(test_loss_list)) + \
                     'RMSE       | mean: %0.4f std: %0.4f\n' % (get_mean_std(rmse_list)) + \
                     'MAE        | mean: %0.4f std: %0.4f\n' % (get_mean_std(mae_list)) + \
                     'CSI        | mean: %0.4f std: %0.4f\n' % (get_mean_std(csi_list)) + \
                     'POD        | mean: %0.4f std: %0.4f\n' % (get_mean_std(pod_list)) + \
                     'FAR        | mean: %0.4f std: %0.4f\n' % (get_mean_std(far_list))+\
                     'R2         | mean: %0.4f std: %0.4f\n' % (get_mean_std(r2_list))+\
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
