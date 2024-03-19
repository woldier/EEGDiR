"""
Author: wolider wong
Date: 2024-1-13
Description: 训练, 测试过程
"""
import yaml
import importlib  # import model
import torch
from datasets import Dataset
import os


def get_config(path: str):
    """
    加载yaml 配置文件
    :param path:
    :return:
    """
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def init_model(conf: dict) -> torch.nn.Module:
    m = _get_model(conf)
    w_path = conf["model"]["weight_path"]
    if w_path is not None and w_path != '':
        m.load_state_dict(torch.load(w_path))
    torch.cuda.empty_cache()
    return m


def _get_model(conf: dict) -> torch.nn.Module:
    """
    加载model
    :return:
    """
    model_path = conf["model"]["path"]
    model_name = conf["model"]["class_name"]
    m = importlib.import_module(model_path)
    clz = getattr(m, model_name)
    return clz(**conf["model"]["config"])  # 实例化对象


def load_dataset(conf: dict, fmt: str = "torch"):
    """

    :param conf: 配置文件
    :param fmt: 数据集中加载的数据的格式
        在datasets 中支持 [None, 'numpy', 'torch', 'tensorflow', 'pandas', 'arrow', 'jax']
        此处仅仅支持 ['numpy', 'torch', 'pandas']
    :return:
    """
    _dataset_fmt_check(fmt)
    train_set = Dataset.load_from_disk(**conf["dataset"]["train"])
    train_set.set_format(fmt)  # set format to pt
    test_set = Dataset.load_from_disk(**conf["dataset"]["test"])
    test_set.set_format(fmt)  # set format to pt
    return train_set, test_set


def _dataset_fmt_check(fmt):
    """
    检查fmt的合法性
    :param fmt:
    :return:
    """
    assert fmt in ['numpy', 'torch', 'pandas'], \
        f'''not support data format! need ['numpy', 'torch', 'pandas'], but have {fmt}'''


def load_dataset_with_path(path: str, fmt: str = "torch"):
    """
    从路径加载数据集
    :param path: 数据集的路径
    :param fmt: 数据格式
    :return:
    """
    _dataset_fmt_check(fmt)
    return Dataset.load_from_disk(path)


# def load_dataset(train_path: str, test_path: str, fmt: str = "torch"):
#     """
#
#     :param train_path: train data path
#     :param test_path: valid data path
#     :param fmt: 数据集中加载的数据的格式
#         在datasets 中支持 [None, 'numpy', 'torch', 'tensorflow', 'pandas', 'arrow', 'jax']
#         此处仅仅支持 ['numpy', 'torch', 'pandas']
#     :return:
#     """
#     assert fmt in ['numpy', 'torch', 'pandas'], \
#         f'''not support data format! need ['numpy', 'torch', 'pandas'], but have {fmt}'''
#     train_set = Dataset.load_from_disk(train_path)
#     train_set.set_format(fmt)  # set format to pt
#     test_set = Dataset.load_from_disk(test_path)
#     test_set.set_format(fmt)  # set format to pt
#     return train_set, test_set


def check_dir(base_path):
    """
    检查输出文件是否存在, 没有的话则创建
    :return:
    """
    if not os.path.exists(base_path):
        os.mkdir(base_path)
        os.mkdir(os.path.join(base_path, 'img'))
        os.mkdir(os.path.join(base_path, 'logs'))
        os.mkdir(os.path.join(base_path, 'weight'))


def config_backpack(conf_path, save_dir):
    config_save_path = save_dir + "config.yml"
    os.system("cp {} {}".format(conf_path, config_save_path))
