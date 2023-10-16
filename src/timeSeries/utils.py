import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import numpy as np
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.layers = nn.ModuleList()
        self.regularized = []

    def add_layer(self, layer, regularized=False):

        self.layers.append(layer)
        self.regularized.append(regularized)

    def remove_loss_layer(self):
        # if isinstance(self.layers[-1], CrossEntropyLossLayer):
        #     del self.layers[-1]
        if isinstance(self.layers[-1], CrossEntropyLoss):
            del self.layers[-1]

    def forward(self, sample):
        layer_input = sample
        for layer_id in range(len(self.layers)):
            # if layer_id == 0:
            layer_input = self.layers[layer_id].forward(layer_input)
            # else:
            #     layer_input = self.layers[layer_id](layer_input)
        return layer_input

    # def backward(self, loss):
    #     l = loss.item()
    #     dL_dlayer_output = l
    #     for layer_id in range(len(self.layers) - 1, -1, -1):
    #         dL_dlayer_output = self.layers[layer_id].backward(dL_dlayer_output)

    def update_params(self):
        for layer_id in range(len(self.layers)):
            self.layers[layer_id].update_params()

    def get_layers(self):
        return self.layers

    def _get_regularized_params(self):
        regularized = []
        for layer_id in range(len(self.layers)):
            if self.regularized[layer_id]:
                regularized.append(self.layers[layer_id].get_params())
        return torch.cat(regularized, axis=1)

def get_centroids_of_segments(data, L, K):
    """

    :param data: the dataset
    :param L: segment length
    :param K: number of centroids
    :return: the top K centroids of the clustered segments
    """
    data_segmented = segment_dataset(data, L)
    centroids = get_centroids(data_segmented, K)
    return centroids

def segment_dataset(data, L):
    """

    :param data:
    :param L: segment length
    :return:
    """
    # number of time series, time series size
    num, length = data.shape
    # number of segments in a time series
    # J = Q - L + 1
    I = int(num*length/L)
    S = np.zeros((I,L))
    # S = np.zeros((J * I, L))
    # create segments
    tmp = data.flatten()
    # tmp = tmp.cpu().numpy()
    for i in range(I):
        S[i,:] = tmp[0:L]
        tmp = tmp[L:] # 取出后弹出队首元素
    # for i in range(S):
    #     for j in range(J):
    #         S[S, :] = data[i, j:j + L]
    return S


def get_centroids(data, k):
    data = TimeSeriesScalerMeanVariance().fit_transform(data)
    # data= TimeSeriesResampler(sz=40).fit_transform(data)
    seed = 0
    np.random.seed(seed)

    dba_km = TimeSeriesKMeans(n_clusters=k, verbose=True, random_state=seed,n_jobs = -1)
    dba_km.fit(data)
    centers = dba_km.cluster_centers_[:,:,-1]

    return centers
