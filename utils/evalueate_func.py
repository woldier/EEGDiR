"""
Author: wolider wong
Date: 2024-1-11
Description: 展示数据
"""
import numpy as np
import torch.nn


def RMS(x: np.ndarray):
    """
    Root Mean Squared (RMS)

    :param x: input
    :return:
    """
    x2 = x ** 2  # x^2
    sum_s2 = np.sum(x2, axis=-1, keepdims=True)  # sum
    return (sum_s2 / x.shape[-1]) ** 0.5


def PSD(x: np.ndarray):
    x = np.fft.fft(x)
    return np.abs(x) ** 2


def CC(x: np.ndarray, f_y: np.ndarray, is_mean: bool = True):
    """
    计算 the correlation coefficient
    Calculate the correlation coefficient
    :param x:
    :param f_y:
    :param is_mean:
        当有多个样本时, 是否要将多个样本的结果取均值, True 代表取均值, False 代表不取均值, 默认为True.
        When there are multiple samples, if or not the results of multiple samples should be averaged,
        True means averaging, False means not averaging, the default is True.
    :return:
    """
    var_f_y = np.var(f_y, axis=-1, keepdims=True)
    var_x = np.var(x, axis=-1, keepdims=True)
    if len(x.shape) == 1:
        cov = np.cov(f_y, x)
        cc = cov[0, 1] / (var_f_y * var_x) ** 0.5
        return cc
    elif len(x.shape) == 2:
        cov = []
        for i in range(x.shape[0]):
            cov_one = np.cov(f_y[i, :], x[i, :])
            cov.append(cov_one[0, 1])
        cc = np.asarray(cov).reshape((-1, 1)) / (var_f_y * var_x) ** 0.5
        if is_mean:
            return cc.mean()
        else:
            return cc


def RRMSE_spectral(x: np.ndarray, f_y: np.ndarray, is_mean: bool = True):
    """
    计算空间尺度上的RRMSE, 可以接受一个 1维度数组, 那么返回值就是该样本的RRMSE
    也可以接受一个batch 的样本, 那么返回值是这一个batch的RRMSE的平均值.
    To compute the RRMSE on a spatial scale, you can accept a 1-dimensional array,
    then the return value is the RRMSE of the sample.
    Or we can accept a batch of samples, then the return value is the average of the RRMSE of the batch.
    :param x:
    :param f_y:
    :param is_mean: 是否要将多个样本的结果取均值, True 代表取均值, False 代表不取均值, 默认为True
        If or not the results of multiple samples should be averaged,
        True means averaging, False means not averaging, default is True.
    :return:
    """
    res = RMS(PSD(f_y) - PSD(x)) / RMS(PSD(x))
    if is_mean:
        return res.mean()
    else:
        return res


def RRMSE_temporal(x: np.ndarray, f_y: np.ndarray, is_mean: bool = True):
    """
    计算时间尺度的RRMSE, 可以接受一个 1维度数组, 那么返回值就是该样本的RRMSE
    也可以接受一个batch 的样本, 那么返回值是这一个batch的RRMSE的平均值
     To calculate the RRMSE of temporal, you can accept a 1-dimensional array,
     then the return value is the RRMSE of the sample.
    Can also accept a batch of samples, then the return value is the average of the RRMSE of the batch.
    :param x: 无噪信号
    :param f_y: 网络预测的无噪声信号
    :param is_mean: 是否要将多个样本的结果取均值, True 代表取均值, False 代表不取均值, 默认为True
    :return:
    """
    res = (RMS(f_y - x) / RMS(x))
    if is_mean:
        return res.mean()
    else:
        return res


def compute_params(net: torch.nn.Module, print_net=False):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    if print_net:
        print(net)
    print('Total number of parameters: %d' % num_params)


def compute_storage_size(model: torch.nn.Module):
    model_size = sum(p.element_size() * p.numel() for p in model.parameters())
    buffer_size = sum(buffer.element_size() * buffer.numel() for buffer in model.buffers())
    total_size = model_size + buffer_size
    print(f"Model Size: {total_size / (1024 ** 2):.2f} MB")