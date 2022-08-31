import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,r2_score,mean_absolute_percentage_error,mean_squared_error

def get_mask(y_true,y_pred):
    """
    原来就有缺失的数据掩膜掉
    :param y_true:
    :param y_pred:
    :return:
    """
    #mask = np.nonzero(y_true)
    mask = np.where(y_true>=1)
    y_pred = y_pred[mask]
    y_true = y_true[mask]
    return y_true,y_pred

def R2(y_true, y_pred):
    '''
    评级函数R2(决定系数)
    '''
    y_true1,y_pred1 = get_mask(y_true,y_pred)

    return r2_score(y_true1,y_pred1)

def MAE(y_true, y_pred):
    '''
    评级函数MAE(平均绝对误差)
    '''
    y_true1, y_pred1 = get_mask(y_true, y_pred)

    return mean_absolute_error(y_true1,y_pred1)

def MAPE(y_true, y_pred,i):
    '''
    评级函数MAPE(平均绝对百分比误差)
    '''
    y_true1, y_pred1 = get_mask(y_true, y_pred)

    return mean_absolute_percentage_error(y_true1,y_pred1)

def RMSE(y_true, y_pred):
    '''
    评级函数RMSE(均方根误差)
    '''
    y_true1, y_pred1 = get_mask(y_true, y_pred)

    return np.sqrt(mean_squared_error(y_true1, y_pred1))

def get_test_result(dataset_name,missing_type,missing_rate,model_name):
    """
    根据索引获取缺失路段数据用于测试
    :param dataset_name:
    :param missing_type:
    :param missing_rate:
    :param model_name:
    :return:
    """
    index_path = "../dataset/{}/{}/test_index{}.csv".format(dataset_name,missing_type,missing_rate)
    test_index_df = pd.read_csv(index_path, header=None, encoding='ANSI')
    test_index = np.mat(test_index_df, dtype=np.int16)

    result_path = "../result/{}/{}_{}{}_result.csv".format(dataset_name,model_name,missing_type,missing_rate)
    result_df = pd.read_csv(result_path, header=None, encoding='ANSI')
    result_all = np.mat(result_df, dtype=np.float32)

    estimation_all = np.mat(result_all[:,:int((result_all.shape[1])/2)])
    real_all = np.mat(result_all[:, int((result_all.shape[1])/2):result_all.shape[1]])

    estimation_test = np.mat(np.zeros((test_index.shape[0], int(result_all.shape[1] / 2))))
    real_test = np.mat(np.zeros((test_index.shape[0], int(result_all.shape[1] / 2))))

    test_index_part = test_index[:,-estimation_all.shape[1]:]
    for i in range(test_index_part.shape[1]):
        estimation_test[:, i] = estimation_all[test_index_part[:, i], i]
        real_test[:,i] = real_all[test_index_part[:, i], i]

    result_test = np.hstack((estimation_test, real_test))

    return result_test

if __name__ == "__main__":
    dataset_name = "sz"
    missing_type = "rm"
    missing_rate = 20
    model_name = "mvgcn"

    result_test = get_test_result(dataset_name=dataset_name,missing_type=missing_type,missing_rate=missing_rate,model_name=model_name)


    estimation = np.array(result_test[:, 0:int((result_test.shape[1])/2)])
    real = np.array(result_test[:, int((result_test.shape[1])/2):result_test.shape[1]])

    evalue1 = np.zeros((estimation.shape[1], 4))
    for i in range(estimation.shape[1]):
        evalue1[i, 1] = MAPE(np.array(real[:, i]), np.array(estimation[:, i]),i)
        evalue1[i, 0] = MAE(np.array(real[:, i]), np.array(estimation[:, i]))
        evalue1[i, 2] = RMSE(np.array(real[:, i]), np.array(estimation[:, i]))
        evalue1[i, 3] = R2(np.array(real[:, i]), np.array(estimation[:, i]))

    mape = np.mean(evalue1[:, 1])
    mae = np.mean(evalue1[:,0])
    rmse = np.mean(evalue1[:,2])
    r2 = np.mean(evalue1[:,3])
    print("Evaluate dataset_name:{}, missing_type:{}, missing_rate:{}".format(dataset_name,missing_type,missing_rate))
    print('rmse:%.4f mae:%.4f mape:%.4f r2:%.4f'%(rmse,mae,mape,r2))


