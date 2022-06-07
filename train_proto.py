from __future__ import print_function, division
import argparse
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
from utils import  load_graph, load_data, load_graph_np
from GNN import GNNLayer
from evaluation import eva, eva_return
from collections import Counter
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from numpy import linalg as LA
import process
from scipy.special import softmax

def propagate(adj, alpha, label, class_num, iter_num):
    if label.shape[1] == 1:
        dense_label = np.zeros([label.shape[0], class_num])
        for i in range(label.shape[0]):
            dense_label[i, label[i, 0]] = 1
    else:
        dense_label = label
    
    H = dense_label
    Z = dense_label
    for i in range(iter_num):
        Z = (1 - alpha) * adj * Z + alpha * H
    Z = softmax(Z, axis=1)
    
    return Z

class DP:
    def __init__(self,X):                    
        self.K = 1
        self.d = X.shape[1]
        self.z = np.mod(np.random.permutation(X.shape[0]),self.K)+1
        self.mu = np.random.standard_normal((self.K, self.d))
        self.sigma = 1
        self.nk = np.zeros(self.K)
        self.pik = np.ones(self.K)/self.K 

        self.mu = np.array([np.mean(X,0)])
        self.Lambda = 0.05
        self.max_iter = 10
        self.obj = np.zeros(self.max_iter)
        self.em_time = np.zeros(self.max_iter)   
        
    def fit(self,X):
        max_iter = self.max_iter        
        [n,d] = np.shape(X)      
        for iter in range(max_iter):
            dist = np.zeros((n,self.K))
            for kk in range(self.K):
                Xm = X - np.tile(self.mu[kk,:],(n,1))
                dist[:,kk] = np.sum(Xm*Xm,1)            
            dmin = np.min(dist,1)
            self.z = np.argmin(dist,1)
            idx = np.where(dmin > self.Lambda)
            
            if (np.size(idx) > 0):
                self.K = self.K + 1
                self.z[idx[0]] = self.K-1 
                self.mu = np.vstack([self.mu,np.mean(X[idx[0],:],0)])                
                Xm = X - np.tile(self.mu[self.K-1,:],(n,1))
                dist = np.hstack([dist, np.array([np.sum(Xm*Xm,1)]).T])
            self.nk = np.zeros(self.K)
            for kk in range(self.K):
                self.nk[kk] = self.z.tolist().count(kk)
                idx = np.where(self.z == kk)
                self.mu[kk,:] = np.mean(X[idx[0],:],0)
            
            self.pik = self.nk/float(np.sum(self.nk))

        return self.z, self.K

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()

        self.fc_1 = nn.Linear(input_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, x):

        h_1 = F.relu(self.fc_1(x))

        h_2 = self.fc_2(h_1)

        return h_2

class simple_encoder(nn.Module):
    def __init__(self, in_features, out_features):
        super(simple_encoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features):
        output = torch.mm(features, self.weight)
        output = F.relu(output)
        return output

class S3CL_Model(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim):
        super(S3CL_Model, self).__init__()
        self.encoder = simple_encoder(in_dim, out_dim)
        self.encoder_momt = simple_encoder(in_dim, out_dim)
        self.projector = MLP(in_dim, hidden_dim, out_dim)
        self.projector_momt = MLP(in_dim, hidden_dim, out_dim)
    @torch.no_grad()
    def _momentum_update(self):
        """
        Momentum update 
        """
        for param_ori, param_momt in zip(self.encoder.parameters(), self.encoder_momt.parameters()):
            param_momt.data = param_momt.data * self.m + param_ori.data * (1. - self.m)
        for param_ori, param_momt in zip(self.projector.parameters(), self.projector_momt.parameters()):
            param_momt.data = param_momt.data * self.m + param_ori.data * (1. - self.m)
    def forward(self, x):
        h = self.encoder(x)
        h_p = self.projector(h)
        with torch.no_grad():  
            self._momentum_update()  
            h_momt = self.encoder_momt(x)
            h_p_momt = self.projector_momt(h_momt)
        return h, h_p, h_p_momt


def evaluation(y, adj, data, model, idx_train, idx_test, out_id=3):
    clf = LogisticRegression(random_state=0, max_iter=2000)
    with torch.no_grad():
        out = model.gae(data, adj)
        embeds = out[out_id]
        train_embs = embeds[idx_train, :] 
        test_embs = embeds[idx_test, :]
        train_labels = torch.Tensor(y[idx_train])
        test_labels = torch.Tensor(y[idx_test])
    clf.fit(train_embs, train_labels)
    pred_test_labels = clf.predict(test_embs)
    return accuracy_score(test_labels, pred_test_labels)


def propagate(adj, alpha, label, class_num, iter_num):
    if label.shape[1] == 1:
        dense_label = np.zeros([label.shape[0], class_num])
        for i in range(label.shape[0]):
            dense_label[i, label[i, 0]] = 1
    else:
        dense_label = label
    
    H = dense_label
    Z = dense_label
    for i in range(iter_num):
        Z = (1 - alpha) * adj * Z + alpha * H
    Z = softmax(Z, axis=1)
    
    return Z

def get_proto_loss(feature, centroid, label_momt, proto_norm):
    
    feature_norm = torch.norm(feature, dim=-1)
    feature = torch.div(feature, feature_norm.unsqueeze(1))
    
    centroid_norm = torch.norm(centroid, dim=-1)
    centroid = torch.div(centroid, centroid_norm.unsqueeze(1))
    
    sim_zc = torch.matmul(feature, centroid.t())
    
    sim_zc_normalized = torch.div(sim_zc, proto_norm)
    sim_zc_normalized = torch.exp(sim_zc_normalized)
    sim_2centroid = torch.gather(sim_zc_normalized, -1, label_momt) 
    sim_sum = torch.sum(sim_zc_normalized, -1, keepdim=True)
    sim_2centroid = torch.div(sim_2centroid, sim_sum)
    loss = torch.mean(sim_2centroid.log())
    loss = -1 * loss
    return loss

def get_proto_norm(feature, centroid, labels):
    num_data = feature.shape[0]
    each_cluster_num = np.zeros([args.n_clusters])
    for i in range(args.n_clusters):
        each_cluster_num[i] = np.sum(labels==i)
    proto_norm_term = np.zeros([args.n_clusters])
    for i in range(args.n_clusters):
        norm_sum = 0
        for j in range(num_data):
            if labels[j] == i:
                norm_sum = norm_sum + LA.norm(feature[j] - centroid[i], 2)
        proto_norm_term[i] = norm_sum / (each_cluster_num[i] * np.log2(each_cluster_num[i] + 10))

    proto_norm_momt = torch.Tensor(proto_norm_term)
    return proto_norm_momt

def train(dataset):
    model = S3CL_Model(args.n_input, 256, 512).to(device)

    model.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))

    optimizer = Adam(model.parameters(), lr=args.lr)

    adj, features, labels, idx_train, idx_val, idx_test = process.load_data(dataset)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    _, adj_np = load_graph_np(args.name, args.k)
    diff = np.load('data/diff_{}_{}.npy'.format(dataset, 0.05), allow_pickle=True)


    features, _ = process.preprocess_features(features)
    features = torch.FloatTensor(features[np.newaxis])
    labels = torch.FloatTensor(labels[np.newaxis])

    norm_adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
    norm_diff = sp.csr_matrix(diff)

    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y

    with torch.no_grad():
        _, _, _, z_momt = model.gae(data, adj)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z_momt.data.cpu().numpy())
    y_pred_last = y_pred
    eva(y, y_pred, 'pae')

    init_labels = kmeans.labels_
    label_momt = torch.Tensor(init_labels).unsqueeze(1)
    label_momt = label_momt.to(torch.int64)
    ori_center = kmeans.cluster_centers_
    centroid_momt = torch.Tensor(ori_center)

    label_kmeans_ori = kmeans.labels_[:, np.newaxis]
    
    with torch.no_grad():
        h, out, out_momt = model(data, adj)

    DP_model = DP(out_momt)
    estimated_K, ps_labels = DP_model.fit(out)
    args.n_clusters = estimated_K
    label_momt = torch.Tensor(ps_labels).unsqueeze(1) 
    centroid_momt = np.dot(ps_labels.T, out_momt) / np.sum(ps_labels.T, axis = 1)[:, np.newaxis]

    label_propagated = propagate(adj_np, 0.1, label_kmeans_ori, args.n_clusters, 10)

    centers_propagated = np.dot(label_propagated.T, z_momt) / np.sum(label_propagated.T, axis = 1)[:, np.newaxis]

    label_propagated_hard = np.argmax(label_propagated, axis=1)
    label_propagated_hard = label_propagated_hard[:, np.newaxis]

    label_momt = torch.Tensor(label_propagated_hard)
    label_momt = label_momt.to(torch.int64)

    proto_norm_momt = get_proto_norm(z_momt, ori_center, label_kmeans_ori)

    _, _, _, idx_train, _, idx_test = process.load_data('citeseer')

    best_acc_clf = 0

    for epoch in range(40):    
        h, out, out_momt = model(data, adj)

        proto_loss = get_proto_loss(out, centroid_momt, label_momt, proto_norm_momt)

        loss =  proto_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        with torch.no_grad():
            h, out, out_momt = model(data, adj)

            classification_acc = evaluation(y, adj, data, model, idx_train, idx_test, 2)

            print('gnn classification accuracy:' + str(classification_acc))

            if classification_acc > best_acc_clf:
                best_acc_clf = classification_acc
            
            kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
            y_pred = kmeans.fit_predict(out_momt.data.cpu().numpy())
            label_kmeans = kmeans.labels_[:, np.newaxis]
            label_propagated = propagate(adj_np, 0.1, label_kmeans, args.n_clusters, 10)
            centers_propagated = np.dot(label_propagated.T, out_momt) / np.sum(label_propagated.T, axis = 1)[:, np.newaxis]
            label_propagated_hard = np.argmax(label_propagated, axis=1)
            label_propagated_hard = label_propagated_hard[:, np.newaxis]
            label_momt = torch.Tensor(label_propagated_hard)
            label_momt = label_momt.to(torch.int64)
            DP_model = DP(out_momt)
            estimated_K, ps_labels = DP_model.fit(X)
            centroid_momt = torch.Tensor(centers_propagated)

            proto_norm_momt = get_proto_norm(out_momt, ori_center, label_kmeans_ori)

    print('Best gnn classification accuracy: ' + str(best_acc_clf))




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='cite')
    parser.add_argument('--k', type=int, default=None)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_clusters', default=1, type=int)
    parser.add_argument('--n_z', default=200, type=int)
    parser.add_argument('--pretrain_path', type=str, default='pkl')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    args.cuda = False

    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    args.pretrain_path = 'data/{}.pkl'.format(args.name + '_new_512_256_512_lr1e-4')
    dataset = load_data(args.name)
    args.lr = 1e-4
    args.n_input = 3703
    args.n_clusters = 1

    print(args)
    train(dataset)

