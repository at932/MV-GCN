import argparse
import numpy as np
from model.mvgcn import *
from utils.input_data import load_data,load_adj_data
import torchmetrics
import os
import torch

def get_data_by_index(data,index):
    """
    根据索引index获取对应数据
    :param data:
    :param index:
    :return:
    """
    data2 = torch.zeros((index.shape[0], index.shape[1])).to(device)
    for i in range(data.shape[0]):
        data2[i, :] = data[i, index[i, :]]
    return data2

def change_outputs_format(data,data_max,data_min):
    """
    根据标准化保存的参数[最大值和最小值]逆标准化处理
    :param data:
    :param data_max:
    :param data_min:
    :return:
    """
    data2 = data*(data_max - data_min)+data_min
    return data2

def get_original_no_missing_data(pred,label):
    """
    掩膜掉原始数据中本身就缺失的（等于0或小于等于1的），避免影响精度指标计算
    :param pred:
    :param lable:
    :return:
    """
    batch_num, road_num = pred.shape
    pred2 = pred.reshape(batch_num * road_num)
    label2 = label.reshape(batch_num * road_num)
    # mask = torch.nonzero(label2)
    mask = torch.where(label2 >= 1)
    pred2 = pred2[mask]
    label2 = label2[mask]
    return pred2,label2

def train(model,dataloader, loss_fun,optimizer, max_val,min_val):
    """
    训练模型
    :param model:
    :param dataloader:
    :param optimizer:
    :param max_val:
    :param min_val:
    :return:
    """
    num_batches = len(dataloader)
    train_loss = 0
    model.train() #开启训练模式
    for batch_idx, data in enumerate(dataloader):
        batch_sparse_speed,  label, index = data
        pred = model(batch_sparse_speed)

        #还原估算结果
        pred1 = change_outputs_format(pred,max_val,min_val)

        #原始数据中本身就缺失的（等于0或小于等于1的）掩膜掉，避免影响精度指标计算
        pred2,label2 = get_original_no_missing_data(pred1,label)

        #损失函数计算
        loss = loss_fun(pred2, label2)

        #反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= num_batches
    return train_loss


def test(model,dataloader, loss_fun,max_val,min_val):
    """
    测试模型
    :param model:
    :param dataloader:
    :param loss_fun:
    :param max_val:
    :param min_val:
    :return:
    """
    test_loss, test_mape, test_mae, test_rmse, test_r2 = 0, 0, 0, 0,0
    num_batches = len(dataloader)

    model.eval() #开启测试模型
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            batch_sparse_speed, label, index = data
            pred = model(batch_sparse_speed)

            # 还原估算结果
            pred1 = change_outputs_format(pred, max_val, min_val)

            # 原始数据中本身就缺失的（等于0或小于等于1的）掩膜掉，避免影响精度指标计算
            pred2, label2 = get_original_no_missing_data(pred1, label)

            #损失函数计算
            loss = loss_fun(pred2, label2)
            #精度指标计算
            mape = torchmetrics.functional.mean_absolute_percentage_error(pred2, label2)
            mae = torchmetrics.functional.mean_absolute_error(pred2,label2)
            rmse = torch.sqrt(torchmetrics.functional.mean_squared_error(pred2, label2))
            r2 = torchmetrics.functional.r2_score(pred2, label2)

            test_loss += loss.item()
            test_mape += mape.item()
            test_mae += mae.item()
            test_rmse += rmse.item()
            test_r2 += r2.item()

    test_loss /= num_batches
    test_mape /= num_batches
    test_mae /= num_batches
    test_rmse /= num_batches
    test_r2 /= num_batches
    return test_loss, test_mape, test_mae, test_rmse, test_r2



def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)

#运行主函数,并保存训练的参数
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000, help='')#迭代参数
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='')#学习率，优化器参数
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='')#衰减率，优化器参数
    parser.add_argument('--batch_size', type=int, default=64, help='')  # 批量数
    parser.add_argument('--missing_rate', type=int, default=20, help='')  # 缺失率，目前有20%、40%和60%
    parser.add_argument('--missing_type', type=str, default='rm', help='')  # 缺失类型，随机缺失rm、块状缺失bm或者混合缺失mm
    parser.add_argument('--dataset_name', type=str, default='sz', help='') #数据集名称,深圳数据集sz或者西安数据集xa

    opt = parser.parse_args()


    epochs = opt.epochs #迭代次数
    learning_rate = opt.learning_rate  # 学习率
    weight_decay = opt.weight_decay  #衰减率
    batch_size = opt.batch_size
    missing_rate = opt.missing_rate  #缺失率
    missing_type = opt.missing_type
    dataset_name = opt.dataset_name

    paths = "./"#数据储存的路径
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    topo_adj, sim_adj = load_adj_data(dataset_name)

    #训练模型
    model = MV_GCN(topo_adj=topo_adj,sim_adj=sim_adj,
                   input_dim=1,
                   hidden1_dim=256,hidden2_dim=128,hidden3_dim=64,
                   hidden4_dim=256,hidden5_dim=128,hidden6_dim=64,
                   output_dim=1)
    model = model.to(device)

    #损失函数
    loss_fun = nn.L1Loss(reduction='mean').to(device)
    #优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_dataset, test_dataset, max_val,min_val = load_data(dataset_name=dataset_name,missing_type=missing_type,missing_rate=missing_rate)

    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    train_dataloader = DeviceDataLoader(train_dataloader, device)
    test_dataloader = DeviceDataLoader(test_dataloader, device)

    train_loss_list, test_loss_list, test_mape_list, test_mae_list, test_rmse_list, test_r2_list, = list(), list(), list(), list(), list(), list()


    for epoch in range(epochs):
        # 训练本模型
        train_loss = train(model, train_dataloader, loss_fun,optimizer, max_val, min_val)

        # 测试本模型
        test_loss, test_mape, test_mae, test_rmse, test_r2 = test(model, test_dataloader,loss_fun, max_val, min_val)
        print('Epoch: %d | train_loss: %.5f  | test_loss: %.5f | test_mape: %.5f | test_mae: %.5f | test_rmse: %.5f | test_r2: %.5f'% (epoch, train_loss, test_loss, test_mape, test_mae, test_rmse, test_r2))

        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        test_mape_list.append(test_mape)
        test_mae_list.append(test_mae)
        test_rmse_list.append(test_rmse)
        test_r2_list.append(test_r2)

    #保存模型参数
    model_savepath = paths + "result/{}/mvgcn_{}{}.pt".format(dataset_name,missing_type,missing_rate)
    torch.save(model.state_dict(), model_savepath)

    print("Done! dataset_name:{}, missing_type:{}, missing_rate:{}".format(dataset_name,missing_type,missing_rate))
    index = test_rmse_list.index(np.min(test_rmse_list))
    print('min_rmse:%.4f' % (test_rmse_list[index]),
          'min_mape:%.4f' % (test_mape_list[index]),
          'min_mae:%.4f' % (test_mae_list[index]),
          'min_r2:%.4f' % (test_r2_list[index]))

    #测试，获得插值结果
    test_dataloader2 = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader2 = DeviceDataLoader(test_dataloader2, device)

    estimation = None
    test_speed = None
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(test_dataloader2):
            inputs, targets, index = data
            pred_result = model(inputs)

            pred_result = change_outputs_format(pred_result,max_val,min_val)
            pred_result = pred_result.detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()

            if (batch_idx == 0):
                estimation = pred_result
                test_speed = targets
            else:
                estimation = np.concatenate((estimation, pred_result), axis=0)
                test_speed = np.concatenate((test_speed, targets), axis=0)

    estimation = estimation.T
    test_speed = test_speed.T

    final_result = np.hstack((estimation, test_speed))

    result_savepath = paths + "result/{}/mvgcn_{}{}_result.csv".format(dataset_name,missing_type,missing_rate)
    np.savetxt(result_savepath, final_result,delimiter=',',fmt="%.6f")


