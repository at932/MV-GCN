# -*- coding: utf-8 -*-
import torch
from torch.nn import functional as F
import torch.nn as nn
from utils.graph_conv import calculate_laplacian_with_self_loop
seed = 200
torch.random.manual_seed(seed)

class MV_GCN(nn.Module):
    def __init__(self,topo_adj,sim_adj,input_dim,hidden1_dim,hidden2_dim,hidden3_dim,hidden4_dim,hidden5_dim,hidden6_dim,output_dim):
        """
        :param topo_adj:邻接矩阵
        :param sim_adj:相似矩阵
        :param input_dim: 输入特征维度=1,即速度
        :param hidden1_dim: 邻接矩阵的GCN的隐藏层维度1
        :param hidden2_dim: 邻接矩阵的GCN的隐藏层维度2
        :param hidden3_dim: 邻接矩阵的GCN的隐藏层维度3
        :param hidden4_dim: 相似矩阵的GCN的隐藏层维度1
        :param hidden5_dim: 相似矩阵的GCN的隐藏层维度2
        :param hidden6_dim: 相似矩阵的GCN的隐藏层维度3
        :param output_dim: 输出维度为1，即速度
        """
        super(MV_GCN,self).__init__()
        self.road_num = topo_adj.shape[0]

        self.weight1 = nn.Parameter(torch.empty(input_dim, hidden1_dim))
        torch.nn.init.kaiming_normal_(self.weight1,mode='fan_in',nonlinearity='relu')

        self.weight2 = nn.Parameter(torch.empty(hidden1_dim,hidden2_dim))
        torch.nn.init.kaiming_normal_(self.weight2,mode='fan_in',nonlinearity='relu')

        self.weight3 = nn.Parameter(torch.empty(hidden2_dim,hidden3_dim))
        torch.nn.init.kaiming_normal_(self.weight3,mode='fan_in',nonlinearity='relu')

        self.weight4 = nn.Parameter(torch.empty(input_dim,hidden4_dim))
        torch.nn.init.kaiming_normal_(self.weight4,mode='fan_in',nonlinearity='relu')

        self.weight5 = nn.Parameter(torch.empty(hidden4_dim,hidden5_dim))
        torch.nn.init.kaiming_normal_(self.weight5,mode='fan_in',nonlinearity='relu')

        self.weight6 = nn.Parameter(torch.empty(hidden5_dim,hidden6_dim))
        torch.nn.init.kaiming_normal_(self.weight6,mode='fan_in',nonlinearity='relu')

        self.bias1 = nn.Parameter(torch.empty(self.road_num, hidden1_dim))
        nn.init.kaiming_normal_(self.bias1, mode='fan_in', nonlinearity='relu')

        self.bias2 = nn.Parameter(torch.empty(self.road_num, hidden2_dim))
        nn.init.kaiming_normal_(self.bias2, mode='fan_in', nonlinearity='relu')

        self.bias3 = nn.Parameter(torch.empty(self.road_num, hidden3_dim))
        nn.init.kaiming_normal_(self.bias3, mode='fan_in', nonlinearity='relu')

        self.bias4 = nn.Parameter(torch.empty(self.road_num, hidden4_dim))
        nn.init.kaiming_normal_(self.bias4, mode='fan_in', nonlinearity='relu')

        self.bias5 = nn.Parameter(torch.empty(self.road_num, hidden5_dim))
        nn.init.kaiming_normal_(self.bias5, mode='fan_in', nonlinearity='relu')

        self.bias6 = nn.Parameter(torch.empty(self.road_num, hidden6_dim))
        nn.init.kaiming_normal_(self.bias6, mode='fan_in', nonlinearity='relu')

        self.topo_adj = topo_adj
        self.sim_adj = sim_adj

        self.input_dim = input_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.hidden3_dim = hidden3_dim
        self.hidden4_dim = hidden4_dim
        self.hidden5_dim = hidden5_dim
        self.hidden6_dim = hidden6_dim
        self.output_dim = output_dim

        self.weight_num = 2
        self.weight_init_value = 1 / self.weight_num
        self.graph_weight = nn.Parameter(torch.FloatTensor(2))
        self.softmax = nn.Softmax(dim=0)
        nn.init.constant_(self.graph_weight, self.weight_init_value)

        self.register_buffer(
            "topo_L", calculate_laplacian_with_self_loop(torch.FloatTensor(self.topo_adj))
        )

        self.register_buffer(
            "sim_L", calculate_laplacian_with_self_loop(torch.FloatTensor(self.sim_adj))
        )

        self.fc1 = nn.Linear(hidden3_dim,output_dim)

    def graph_conv(self,X,weights,bias,input_dim,output_dim,batch_size,num_nodes,L):
        #输入特征 X [batch_size,num_nodes,input_dim]
        X = X.permute(1, 0, 2)  # X [num_nodes,batch_size,input_dim]
        X = X.reshape(num_nodes,-1)#X [num_nodes,batch_size*input_dim]
        X = torch.matmul(L, X)
        X = X.reshape(num_nodes,batch_size,input_dim)
        X = X.permute(1,0,2)#X [batch_size,num_nodes,input_dim]
        X = X.reshape(batch_size * num_nodes, input_dim)
        X = torch.matmul(X, weights)
        X = X.reshape(batch_size, num_nodes, output_dim)
        out = X + bias
        return out

    def forward(self, X):
        w = self.softmax(self.graph_weight)

        batch_size,num_nodes = X.shape
        X= X.unsqueeze(2)

        # 第一层耦合卷积的搭建
        adj_pre1 = self.graph_conv(X,self.weight1,self.bias1,self.input_dim,self.hidden1_dim,batch_size,num_nodes,self.topo_L)
        sim_pre1 = self.graph_conv(X,self.weight4,self.bias4,self.input_dim,self.hidden4_dim,batch_size,num_nodes,self.sim_L)
        pre1 = F.relu(w[0]*adj_pre1 + w[1]*sim_pre1)

        # 第二层耦合卷积的搭建
        adj_pre2 = self.graph_conv(pre1, self.weight2, self.bias2, self.hidden1_dim, self.hidden2_dim, batch_size, num_nodes,self.topo_L)
        sim_pre2 = self.graph_conv(pre1, self.weight5, self.bias5, self.hidden4_dim, self.hidden5_dim, batch_size, num_nodes,self.sim_L)
        pre2 = F.relu(w[0]*adj_pre2 + w[1]*sim_pre2)

        # 第三层耦合卷积的搭建
        adj_pre3 = self.graph_conv(pre2, self.weight3, self.bias3, self.hidden2_dim, self.hidden3_dim, batch_size,num_nodes, self.topo_L)
        sim_pre3 = self.graph_conv(pre2, self.weight6, self.bias6, self.hidden5_dim, self.hidden6_dim, batch_size,num_nodes, self.sim_L)
        pre3 = F.relu(w[0]*adj_pre3 + w[1]*sim_pre3)

        pre = self.fc1(pre3)  # 全连接层
        pre = pre.squeeze(2)
        return pre