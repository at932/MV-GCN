import numpy as np
import pandas as pd
import torch

def load_sparse_speed_data(path,missing_type,missing_rate,dtype = np.float32):
    filepath = path + "{}/sparse_speed{}.csv".format(missing_type,missing_rate)
    sparse_speed_data_df = pd.read_csv(filepath,header=None,encoding='ANSI')
    sparse_speed_data = np.array(sparse_speed_data_df,dtype=dtype)
    return sparse_speed_data

def load_train_index(path,missing_type,missing_rate,dtype = np.int16):
    filepath = path + "{}/train_index{}.csv".format(missing_type, missing_rate)
    train_index_df = pd.read_csv(filepath, header=None, encoding='ANSI')
    train_index = np.array(train_index_df, dtype=dtype)
    return train_index

def load_test_index(path,missing_type,missing_rate,dtype = np.int16):
    filepath = path + "{}/test_index{}.csv".format(missing_type, missing_rate)
    test_index_df = pd.read_csv(filepath, header=None, encoding='ANSI')
    test_index = np.array(test_index_df, dtype=dtype)
    return test_index

def load_speed_data(path,dataset_name,dtype = np.float32):
    filepath = path + "speed.csv"
    speed_data_df = pd.read_csv(filepath,header=None,encoding='ANSI')
    speed_data = np.array(speed_data_df,dtype=dtype)
    if dataset_name == "sz":
        new_speed_data = np.hstack((speed_data, speed_data, speed_data, speed_data, speed_data,
                                    speed_data, speed_data, speed_data, speed_data, speed_data)) #sz数据集中，一个时间戳下的样本重复生成10个模拟缺失实例，以扩充样本数量
    elif dataset_name == "xa":
        new_speed_data = speed_data
    return new_speed_data

def generate_dataset(sparse_speed_data, speed_data, train_index, test_index,ratio):
    sample_num = sparse_speed_data.shape[1]
    train_num = int(sample_num*ratio)
    test_num = sample_num-train_num
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(sparse_speed_data[:,0:train_num].T), torch.FloatTensor(speed_data[:,0:train_num].T),torch.LongTensor(train_index[:,0:train_num].T)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(sparse_speed_data[:,train_num:sample_num].T), torch.FloatTensor(speed_data[:,train_num:sample_num].T),torch.LongTensor(test_index[:,train_num:sample_num].T)
    )

    return train_dataset, test_dataset

def load_data(dataset_name,missing_type,missing_rate):
    path = "./dataset/{}/".format(dataset_name)
    ratio = 0.8 #划分测试集和训练集

    sparse_speed_data = load_sparse_speed_data(path, missing_type,missing_rate)
    train_index = load_train_index(path, missing_type,missing_rate)
    test_index = load_test_index(path, missing_type,missing_rate)
    speed_data = load_speed_data(path,dataset_name)

    #标准化
    max_val = np.max(speed_data)
    min_val = 0
    norm_sparse_speed_data = (sparse_speed_data-min_val)/(max_val-min_val)

    train_dataset, test_dataset = generate_dataset(norm_sparse_speed_data,speed_data,train_index, test_index,ratio)

    return train_dataset,test_dataset,max_val,min_val

def load_adj(path,adj_name,dtype = np.float32):
    filepath = path + "{}.csv".format(adj_name)
    adj_df = pd.read_csv(filepath, header=None, encoding='ANSI')
    adj = np.array(adj_df, dtype=dtype)
    return adj

def load_adj_data(dataset_name):
    path = r"./dataset/{}/".format(dataset_name)
    topo_adj = load_adj(path,"topo_adj")
    sim_adj = load_adj(path,"sim_adj")
    return topo_adj, sim_adj
