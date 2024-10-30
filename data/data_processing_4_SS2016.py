# -*- coding: utf-8 -*-
"""
@Time ： 3/26/24 3:32 AM
@Auth ： woldier wong
@File ：data_prepare.py
@IDE ：PyCharm
@DESCRIPTION：新的数据
"""
import scipy.io as sio
from utils.data_show import show_img
import numpy as np
import matplotlib.pyplot as plt
from datasets import Dataset




def RMS(x: np.ndarray):
    """
    Root Mean Squared (RMS)

    :param x: input
    :return:
    """
    x2 = x ** 2  # x^2
    sum_s2 = np.sum(x2, axis=-1, keepdims=True)  # sum
    return (sum_s2 / x.shape[-1]) ** 0.5


def split_data(data, segment_length=512):
    """
    将数据拆分成512的小段
    :param data:
    :param segment_length:
    :return:
    """
    segments = []
    for sample in data:
        sample_length = len(sample)
        if sample_length < segment_length:
            continue  # 如果样本长度不足512，则丢弃该样本
        num_segments = sample_length // segment_length
        for i in range(num_segments):
            start_idx = i * segment_length
            end_idx = start_idx + segment_length
            segment = sample[start_idx:end_idx]
            segments.append(segment)
    return segments


def compute_noise_signal(x: np.ndarray, n: np.ndarray, snr: np.ndarray):
    """
    λ is a hyperparameter to control the signal-to-noise ratio (SNR) in the contaminated EEG signal y
    SNR = 10 log( RMS(x) / RMS(λ · n) )

    SNR = 10 log ( RMS(x) / ( λ · RMS(n) )  )

    (SNR / 10 ) ** 10 = RMS(x) / ( λ · RMS(n) )

    y = x + λ · n
    :param x: noise-free signal
    :param n: noise signal
    :param snr:
    :return:
    """
    lamda = RMS(x) / ((10 ** (snr / 10)) * RMS(n))
    return x + lamda * n


def normalize(x: np.ndarray, y: np.ndarray, mean_norm=False):
    """
    In order to facilitate the learning procedure, we normalized the input contaminated EEG segment and the ground-truth
    EEG segment by dividing the standard deviation of contaminated EEG segment according to
    x_bar = x / std(y)
    y_bar = y / std(y)
    :param x: noise-free signal
    :param y: contaminated signal
    :param mean_norm: bool , default false  . If true, will norm mean to 0
    :return:
    """
    mean = y.mean() if mean_norm else 0
    std = y.std(axis=-1, keepdims=True)
    x_bar = (x - mean) / std
    y_bar = (y - mean) / std
    return x_bar, y_bar, std


def generate_signal_pair(x: np.ndarray, n: np.ndarray, snr_start=-7, snr_end=2):
    """
    生成数据样本对, 初始样本将会混入, 不同噪声水平的噪声形成噪声信号. 最后我们将样本对都进行了方差norm
    :param x: 噪声信号
    :param n: 要混入的噪声信号
    :param snr_start: 信噪比的开始值
    :param snr_end:  信噪比的结束值
    :return:
    """
    nums = x.shape[0]  # 得到数据量
    # 需要先区分开训练集和测试集, 因为后面做了重复后再挑选的话, 那么有可能所有的无噪声样本都出现在了训练集中
    # 引入信噪比
    step = snr_end - snr_start + 1
    snr_table = np.linspace(snr_start, snr_end, step)
    snr_table = snr_table.reshape((1, snr_table.shape[0]))  # reshape to [1, 10]
    nums_table = np.zeros((nums, 1))  # reshape to [nums, 1]
    snr_table = snr_table + nums_table  # broadcast to [nums, 10]
    snr_table = snr_table.reshape((-1, 1))  # match samples
    x = np.repeat(x, step, axis=0)  # 每个样本重复 step 次, 以便与不同噪声水平的噪声相加
    # TODO 这里还可以shuffle n , 理论上讲应该会形成更广泛的样本对, 但是尚不清楚这样做的网络训练结果
    n = np.repeat(n, step, axis=0)  # 重复 step 次与不同的信噪比做对应
    y = compute_noise_signal(x, n, snr_table)
    # normalize
    x_bar, y_bar, std = normalize(x, y)
    return x_bar, y_bar, std


def main():
    # 获取数据
    x_all, y_all = load_data()
    # 统计snr
    x_all, y_all = statistic_snr(x_all, y_all)

    # 随机分配元素到N个子集
    res_dict = generate_dataset(x_all, y_all)
    # return res_dict
    train_dataset_eog = Dataset.from_dict({"x": res_dict["train"]["x"],
                                           "y": res_dict["train"]["y"],
                                           "std": res_dict["train"]["std"],
                                           })
    test_dataset_eog = Dataset.from_dict({"x": res_dict["valid"]["x"],
                                          "y": res_dict["valid"]["y"],
                                          "std": res_dict["valid"]["std"],
                                          })
    # 保存数据集到文件
    train_dataset_eog.save_to_disk("../dataset/SS-EOG/train")
    test_dataset_eog.save_to_disk("../dataset/SS-EOG/test")


def generate_dataset(x_all, y_all):
    dicts = {"train": 0.8, "valid": 0.2}
    split = list(dicts.values())
    random_subsets = np.random.choice(len(split), size=len(x_all), p=split)
    # 将数组分配到N个子集中
    res_dict = {}
    for i, name in enumerate(dicts.keys()):
        # EEDDenoisingNet 中的生成方式
        eeg_split, artifact_split = x_all[random_subsets == i], y_all[random_subsets == i] - x_all[random_subsets == i]
        x, y, std = generate_signal_pair(eeg_split, artifact_split)
        # 不适用 fix SNR
        # eeg_split, eeg_split_noised = x_all[random_subsets == i], y_all[random_subsets == i]
        # x, y, std = normalize(eeg_split, eeg_split_noised)
        res_dict[name] = {"x": x, "y": y, "std": std}
    idx = 0
    show_img(res_dict["train"]["x"][idx*10:idx*10+10] * res_dict["train"]["std"][idx*10:idx*10+10],
             res_dict["train"]["y"][idx*10:idx*10+10] * res_dict["train"]["std"][idx*10:idx*10+10],
             None, save_name='./img-processed.svg',
             format='svg', dpi=50,
             bbox_inches='tight')
    return res_dict


def statistic_snr(x_all, y_all):
    """
     SNR = 10 log( RMS(x) / RMS(λ · n) )
    """
    SNR = 10 * np.log10(RMS(x_all) / (RMS(y_all - x_all)))
    # show_img(
    #     y['sim1_con'][0][0:512:512],
    #     x['sim1_resampled'][0][0:512],
    #     None, save_name='./img.svg',
    #     format='svg', dpi=50,
    #     bbox_inches='tight')
    # 取整 SNR 值
    rounded_snr_values = np.round(SNR).astype(int)
    # 统计各 SNR 值的样本个数
    unique_snr_values, counts = np.unique(rounded_snr_values, return_counts=True)
    # 可视化
    colors = ['#de3024', '#fc8c59', '#fdde90', '#e7f2f3', '#4b74b1']
    plt.bar(unique_snr_values, counts, color=colors)
    plt.xlabel('SNR')
    plt.ylabel('sample num')
    plt.title('Number of samples for each SNR value')
    plt.savefig("origin_snr_count.svg", format='svg', dpi=50, bbox_inches='tight')
    selected_indices = np.where(SNR < 5)[0]  # mention in paper https://github.com/woldier/EEGDiR/issues/2
    return x_all[selected_indices], y_all[selected_indices]

def load_data():
    y_row = sio.loadmat('./source/Contaminated_Data.mat')
    x_row = sio.loadmat('./source/Pure_Data.mat')
    x_all, y_all = [], []
    for x_key, y_key in zip(x_row.keys(), y_row.keys()):
        if x_key in ['__header__', '__version__', '__globals__']:
            continue
        assert x_key[:-10] == y_key[:-4], "the colum name not equal!"
        x_colum, y_colum = x_row[x_key], y_row[y_key]
        x_colum_split = split_data(x_colum)
        y_colum_split = split_data(y_colum)

        x_all.extend(x_colum_split)
        y_all.extend(y_colum_split)
    # show_img(np.asarray(x_all[:8]),
    #          np.asarray(y_all[:8]),
    #          None, save_name='./img.svg',
    #          format='svg', dpi=50,
    #          bbox_inches='tight')
    ##  viz
    # from utils.data_show import show_x_y
    # for idx in range(16):
    #     x_bar_list = []  # 存放结果
    #     legend_names = []  # 存放名称
    #     x_axis = np.linspace(0, 2.56, 512)
    #     x_bar_list.append(x_all[idx])
    #     legend_names.append("noise-free signal")
    #     x_bar_list.append(y_all[idx])
    #     legend_names.append("noise signal")
    #     data = np.asarray(x_bar_list)
    #     show_x_y(
    #         x=x_axis,
    #         y=data,
    #         legend_names=legend_names,
    #         title="",
    #         x_label="Time(s)",
    #         # y_label="",
    #         y_label="Amplitude(uV)",
    #         figsize=(6, 3),
    #         save_name="./origin/origin-{}.svg".format(idx),
    #         format='svg', dpi=50, bbox_inches='tight'  # additional args for savefig
    #     )
    x_all, y_all = np.asarray(x_all), np.asarray(y_all)
    return x_all, y_all


if __name__ == '__main__':
    main()
