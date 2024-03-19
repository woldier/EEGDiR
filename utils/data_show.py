"""
Author: wolider wong
Date: 2024-1-11
Description: show image
"""
import numpy as np
from typing import Optional, Union
from matplotlib import pyplot as plt
import os


def show_img(x: np.ndarray, y: np.ndarray, pre: Optional[np.ndarray], save_name=None, **kwargs):
    """
    展示图片

    :param x: noise-free data
    :param y:  noise data
    :param pre: predict data
    :param save_name:
    :return:
    """
    if len(x.shape) == 1:
        show_one(x, y, pre, save_name, **kwargs)
    elif len(x.shape) == 2:
        show_batch(x, y, pre, save_name, **kwargs)
    else:
        raise AttributeError("not support shape ...")


def show_batch(x: np.ndarray, y: np.ndarray, pre: Optional[np.ndarray], save_name: Optional[str],
               max_batch_size: int = 32, **kwargs):
    """
    show a batch of image
    :param pre:
    :param save_name:
    :param x:
    :param y:
    :param max_batch_size: max image number
    :return:
    """
    if x.shape[0] > max_batch_size:  # crop batch
        x = x[0:max_batch_size]
    num = x.shape[0]
    fig = plt.figure(figsize=(16, ((num + 1) // 2) * 4))
    for i in range(num):
        ax = plt.subplot(((num + 1) // 2), 2, i + 1)
        if y is not None:
            plt.plot(y[i].squeeze(), label="noise signal")
        if x is not None:
            plt.plot(x[i].squeeze(), label="signal")
        if pre is not None:
            plt.plot(pre[i].squeeze(), label="pre signal", ls='-.')
        plt.legend()
    if save_name is None:
        plt.show()
    else:
        check_dir(save_name)
        plt.savefig(save_name, **kwargs)


def show_one(x: np.ndarray, y: np.ndarray, pre: Optional[np.ndarray], save_name: Optional[str], **kwargs):
    """
    展示一张
    :param pre:
    :param save_name:
    :param x:
    :param y:
    :return:
    """
    plt.figure(figsize=(9, 6), dpi=80)
    if x is not None:
        plt.plot(x.squeeze(), label="signal")
    if y is not None:
        plt.plot(y.squeeze(), label="noise signal")
    if pre is not None:
        plt.plot(pre.squeeze(), label="pre signal", ls='-.')
    if save_name is None:
        plt.show()
    else:
        check_dir(save_name)
        plt.savefig(save_name, **kwargs)


def show_x_y(x: np.ndarray,
             y: Union[np.ndarray, list],
             legend_names: list,
             title: str,
             x_label: str,
             y_label: str,
             save_name: Optional[str] = None,
             figsize: tuple = (12, 8),
             grid: bool = True,
             **kwargs):
    """
    展示x,y 图. x轴接受一个移位数组, y 可以接受一个二维的数组[y1,y2,y3] shape is [num_curves, num_points]

    Example:
    y1 = [i for i in range(10)]
    y2 = [i + 2 for i in range(10)]
    y3 = [i + 3 for i in range(10)]
    y = np.stack([y1, y2, y3])
    show_x_y(
    x=np.asarray([i for i in range(10)]),
    y=y, # or direct use y = [y1, y2, y3]
    legend_names=["curves1", "curves2", "curves3"],
    title="test-multiY",
    x_label="x-label",
    y_label="y-label",
    figsize=(8, 8),
    save_name="./test.svg",
    format='svg', dpi=50,  # additional args for savefig
    color=['blue', 'green', 'red'], linestyle=['-', '--', '-.'],marker=['o', 's', '^']
    )


    :param x: x axis
    :param y:  y axis 可以接受二维数组 [y1,y2,y3]
    :param legend_names: 每条曲线的名字 example ["curves1","curves2","curves3"]
    :param title: 图像的title
    :param x_label: x轴的名称
    :param y_label: y轴的名称
    :param save_name:  保存位置的path
    :param figsize: 图像的大小形状
    :param grid: 是否展示网格
    :param kwargs: save 图像时送给plt.savefig(save_name, **kwargs) 的额外参数, 例如 format='svg', dpi=50
    :return:
    """
    plt.figure(figsize=figsize)
    if isinstance(y, list):
        y = np.stack(y)
    # 在 plt.plot(x, y.T) 中, y.T 是对 y 进行转置操作.
    # 这是因为 y 的形状为 (num_curves, num_points),
    # 其中 num_curves 是曲线的数量, num_points 是每条曲线上的点的数量.
    # 在默认的情况下, plt.plot 期望每一列对应一条曲线,因此需要对 y进行转置.
    plt.plot(x, y.T)
    # 添加网格线
    if grid:
        plt.grid(linestyle='--', )
        # plt.grid(True, linestyle='--', )
    # 添加图例
    plt.legend(legend_names)
    # 添加标题和轴标签
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if save_name is None:
        plt.show()
    else:
        check_dir(save_name)
        plt.savefig(save_name, **kwargs)


def check_dir(path: str):
    """
    检查保存文件所在的文件夹是否存在
    :param path:
    :return:
    """
    # 获取文件所在文件夹的路径
    folder_path = os.path.dirname(path)
    # 检查文件夹是否存在，如果不存在则创建
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
