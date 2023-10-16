import numpy as np
import torch
from dtw import dtw
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import cosine_similarity, paired_distances
from sklearn import preprocessing
import torch.nn as nn
from tslearn import metrics
from numba import njit, prange
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

class SoftMinLayer(nn.Module):
    def __init__(self,args, sequence, learning_rate=0.01, alpha=-100 ):
        super(SoftMinLayer, self).__init__()
        """

        :type alpha:
        :param sequence:
        :param alpha:
        """
        self.S = sequence.flatten()
        # self.L = np.size(sequence, 1)
        self.L = len(self.S)
        self.alpha = alpha
        # learning rate
        self.eta = learning_rate
        # layer input holder
        self.T = None
        # layer output holder
        self.current_output = None
        # derivative of Loss w.r.t. shapelet values
        self.dL_dS = None
        # holder of pre-calculated values to speed up the calculations
        self.J = None  # number of segments in input time-series
        self.D = None  # (1 X J) distances between shapelet and the current time-series segments
        self.xi = None  # (1 X J)
        self.psi = None
        self.M = None  # soft minimum distance
        self.device = args.device
        self.mask = self.mask_mat()
        self.args = args#torch.zeros((4, 4)) #self.mask_mat()  #mask = torch.zeros((self.L, l2))

    def forward(self, layer_input):
        self.T = layer_input
        # print("testeststyadfuyfgdu:",self.S)
        if self.args.data.staticgraph ==True:
            self.M = torch.cosine_similarity(self.S, self.T, dim=0).to(self.device)
        else:
            self.M = self.dist_soft_min()

        return self.M

    # def backward(self, dL_dout):
    #     """
    #
    #     :param dL_dout:
    #     :return: dL_dS (1 X self.L)
    #     """
    #     # (1 X J): derivative of M (soft minimum) w.r.t D_j (distance between shapelet and the segment j of the
    #     # time-series)
    #     T = self.T[0]
    #     dM_dD = self.xi * (1 + self.alpha * (self.D - self.M)) / self.psi
    #     # (J X L) : derivative of D_j w.r.t. S_l (shapelet value at position l)
    #     dD_dS = torch.zeros((self.J, self.L))
    #     for j in range(self.J):
    #         dD_dS[j, :] = 2 * (self.S - T[j:j + self.L]) / self.L
    #     # (1 X L) : derivative of M w.r.t. S_l
    #     dM_dS = np.dot(dM_dD, dD_dS)
    #     # (1 X L) : derivative of L w.r.t S_l. Note dL_dout is dL_dM
    #     self.dL_dS = dL_dout * dM_dS
    #     return self.dL_dS

    # def dist_soft_min(self,mask):
    #     Q = self.T.numel()
    #     # self.J = int(Q/self.L)
    #     self.J = Q - self.L
    #     M_numerator = 0
    #     # for each segment of T
    #     self.D = []
    #     self.xi = []
    #     self.psi = 0
    #     for j in range(0, self.J, int(self.L / 2)):
    #         # d = self.dist_sqr_error(self.T[j:j + self.L])
    #         # d = self.cosine_dst(self.T[j:j + self.L])
    #         d = self.fast_dtw_dist(self.T[ j:j + self.L],mask)
    #         self.D.append(d)
    #         xi = torch.exp(self.alpha * d)
    #         self.xi.append(xi)
    #         M_numerator += d * xi
    #         self.psi += xi
    #     M = M_numerator / self.psi
    # # M = self.psi/self.J
    # #     M = min(self.D)
    #     return M
    def dist_soft_min(self):
        Q = self.T.numel()
        self.J = Q - self.L
        M_numerator = 0
        # for each segment of T
        self.D = []
        self.xi = []
        self.psi = 0
        it = 0
        for j in range(0, self.J, int(self.L / 2)):
            test = self.T[j:j + self.L]
            d = self.fast_dtw_dist(self.T[j:j + self.L])
            self.D.append(d)
            xi = torch.exp(self.alpha * d)
            self.xi.append(xi)
            M_numerator += d * xi
            self.psi += xi
            it+=1
        # M = M_numerator / self.psi*self.alpha
        M = self.psi/it
        return M
    #
    # def soft_jacard_dis(self):
    #     Q = self.T.size
    #     self.J = int(Q/self.L)
    #     M_numerator = []
    #     # for each segment of T
    #     self.D = np.zeros((1, self.J))
    #     self.xi = np.zeros((1, self.J))
    #     self.psi = 0
    #     x = self.T.reshape(-1,self.L)
    #
    #     x = torch.from_numpy(x)
    #     y = torch.from_numpy(self.S)
    #     d = self.L
    #     t = self.alpha
    #
    #     x_soft_max = torch.softmax(t * x.float(), dim=1) * x
    #     y_soft_max = torch.softmax(t * y.float(), dim=1) * y
    #
    #     x_soft_min = torch.softmax(-t * x.float(), dim=1) * x
    #     y_soft_min = torch.softmax(-t * y.float(), dim=1) * y
    #     # 这一步用矩阵进行计算，利用a+b = ab-(a-1)(b-1)+1的原理，如下图所示
    #     max_matrix = x_soft_max.mm(y_soft_max.t()) - (x_soft_max - 1).mm((y_soft_max - 1).t())+d
    #     min_matrix = x_soft_min.mm(y_soft_min.t()) - (x_soft_min - 1).mm((y_soft_min - 1).t())+d
    #
    #     dist = min_matrix / max_matrix
    #     dist = (1. - dist).numpy()
    #     return dist
    #
    #
    # def dist_sqr_error(self, T_j):
    #     """
    #
    #     :param T:
    #     :return:
    #     """
    #     x =  T_j.reshape(1, -1)
    #     y = torch.from_numpy(self.S.reshape(1, -1)).to(self.device)
    #     dist = torch.pow((x - y) ,2)
    #     dist = torch.sum(dist) / self.L
    #
    #     return dist
    # def cosine_dst(self,ts):
    #     x = ts.reshape(1, -1).cpu().numpy()
    #     y = self.S.reshape(1, -1)
    #     dist = paired_distances(x, y, metric='cosine')
    #     dist = torch.tensor(dist).to(self.device)
    #     return dist
    #
    # def dist_dtw(self):
    #     x = self.S.reshape(-1, 1)
    #     y = self.T.reshape(-1, 1)
    #     manhattan_distance = lambda x, y: np.abs(x - y)
    #     d, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=manhattan_distance)
    #     return d

    def fast_dtw_dist(self,ts):
        test = int(len(ts)/self.args.timeseries.shaplet_segment)
        x = ts.reshape(-1,int(len(ts)/self.args.timeseries.shaplet_segment))
        y = self.S.reshape(-1, int(len(self.S)/self.args.timeseries.shaplet_segment))
        # dist, path = fastdtw(x, y, dist=euclidean)
        # path, sim = metrics.dtw_path(x, y)
        sim = self.DTW_pytorch(x,y)
        # alignment, sim = metrics.soft_dtw_alignment(x, y, gamma=gamma)
        sim = sim.to(self.device)
        return sim

    def dist_fast_dtw(self):
        x = self.T.reshape(4, -1).cpu().numpy()
        y = self.S.reshape(4, -1)
        dist, path = fastdtw(x, y, dist=euclidean)
        path, sim = metrics.dtw_path(x, y)
        dist = torch.tensor(dist).to(self.device)
        dist = -np.exp(self.alpha * dist) - 1
        return dist

    def get_params(self):
        """

        :return:
        """
        return self.S

    def set_params(self, param):
        """

        :param param:
        :return:
        """
        self.S = param

    def update_params(self):
        self.S -= self.eta * self.dL_dS

    def get_size(self):
        return self.L

    def get_shapelet(self):
        return self.S.ravel().cpu().numpy().tolist()

    def mask_mat(self):
        l1 = 5
        # l2 = s2.shape[0]
        radius = l1 // 5
        mask = torch.full((l1, l1), torch.inf)
        width = radius
        for i in prange(l1):
            lower = max(0, i - radius)
            upper = min(l1, i + width) + 1
            mask[i, lower:upper] = 0.
        return mask


    def DTW_pytorch(self,s1,s2):
        # s1 = s1.float()
        # s2 = s2.float()
        if len(s1) == 0 or len(s2) == 0:
            raise ValueError("One of the input time series contains only nans or has zero length.")
        l1 = s1.shape[0]
        l2 = s2.shape[0]
        # mask
        # mask = torch.zeros((l1, l2))

        """Compute the accumulated cost matrix score between two time series."""
        cum_sum = torch.full((l1 + 1, l2 + 1), torch.inf)
        cum_sum[0, 0] = 0.
        for i in range(l1):
            for j in range(l2):
                if torch.isfinite(self.mask[i, j]):
                    cum_sum[i + 1, j + 1] = torch.sum(torch.square(s1[i]- s2[j]))
                    cum_sum[i + 1, j + 1] += min(cum_sum[i, j + 1],
                                                 cum_sum[i + 1, j],
                                                 cum_sum[i, j])
        acc_cost_mat = cum_sum[1:, 1:]
        dst = torch.sqrt(acc_cost_mat[-1, -1])
        return dst
    # path = _return_path(acc_cost_mat)

def _local_squared_dist(x, y):
    dist = torch.sum(torch.square(x-y))
    # dist = 0.

    # for di in range(x.shape[0]):
    #     diff = x[di] - y[di]
    #     dist += torch.square(diff)
    return dist