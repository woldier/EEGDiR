"""
Author: wolider wong
Date: 2024-1-11
Description: EEG, EOG, EMG 数据预处理, 形成用于训练的数据集
cite: EEGdenoiseNet: A benchmark dataset for end-to-end deep learning solutions of EEG denoising
https://arxiv.org/abs/2009.11662
"""
import numpy as np
from datasets import Dataset
from utils.data_show import show_img


def RMS(x: np.ndarray):
    """
    Root Mean Squared (RMS)

    :param x: input
    :return:
    """
    x2 = x ** 2  # x^2
    sum_s2 = np.sum(x2, axis=-1, keepdims=True)  # sum
    return (sum_s2 / x.shape[-1]) ** 0.5


def nums_match(eeg: np.ndarray, artifact: np.ndarray):
    """
    match the EEG samples nums with EMG or EOG.
    because EEG size is 4514, EMG size is 3400, EOG size 5598.
    for EMG , we  use 3400 EEG segments and 3400 ocular artifact segments
    for EOG, we randomly reuse some EEG segments.
    :param eeg: eeg samples
    :param artifact:  ocular or myogenic artifact contaminated signals
    :return:
    """
    eeg_nums = eeg.shape[0]
    artifact_nums = artifact.shape[0]
    # np.random.seed(109)
    if eeg_nums > artifact_nums:
        # 裁剪数据集
        # 从eeg中挑选一部分索引, 由于没有使用到所有的数据, 因此我们从EEG中随机取
        select_index = np.random.choice(eeg_nums, artifact_nums, replace=False)
        return eeg[select_index, :], artifact
    elif eeg_nums < artifact_nums:
        # reuse eeg
        select_index = np.random.choice(eeg_nums, artifact_nums - eeg_nums, replace=False)  # 从eeg中挑选一部分索引
        eeg_reused = eeg[select_index, :]
        eeg_cat = np.concatenate((eeg, eeg_reused), axis=0)  # cat
        return eeg_cat, artifact


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


def generate_dataset(eeg: np.ndarray, artifact: np.ndarray, dicts=None):
    if dicts is None:
        dicts = {"train": 0.8, "valid": 0.1, "test": 0.1}
    split = list(dicts.values())
    assert int(sum(split)) == 1, "all the subset sum must be 1"
    # 随机分配元素到N个子集
    random_subsets = np.random.choice(len(split), size=len(eeg), p=split)
    # 将数组分配到N个子集中
    res_dict = {}
    for i, name in enumerate(dicts.keys()):
        eeg_split, artifact_split = eeg[random_subsets == i], artifact[random_subsets == i]
        x, y, std = generate_signal_pair(eeg_split, artifact_split)
        res_dict[name] = {"x": x, "y": y, "std": std}
    return res_dict


def init_dataset():
    EEG = np.load("./source/EEG_all_epochs.npy")
    """
    To match the number of EEG segments with myogenic artifact segments, we randomly reuse some EEG segments.
    """
    EMG = np.load("./source/EMG_all_epochs.npy")
    """
    The semi-synthetic ocular artifact contaminated signals are from 3400 EEG segments and 3400 ocular artifact segments
    , with 80% for generating the training set, 10% for generating the
    validation set, and 10% for generating the test set
    """
    EOG = np.load("./source/EOG_all_epochs.npy")
    np.random.seed(109)
    """==========================================mix EEG with EOG===================================================="""
    # reuse EEG
    eeg, eog = nums_match(EEG, EOG)
    # split to train valid
    # 定义分成N个子集的概率，这里设置为
    res_eog = generate_dataset(eeg, eog, dicts={"train": 0.8, "valid": 0.2})
    train_dataset_eog = Dataset.from_dict({"x": res_eog["train"]["x"],
                                           "y": res_eog["train"]["y"],
                                           "std": res_eog["train"]["std"],
                                           })
    test_dataset_eog = Dataset.from_dict({"x": res_eog["valid"]["x"],
                                          "y": res_eog["valid"]["y"],
                                          "std": res_eog["valid"]["std"],
                                          })
    # 保存数据集到文件
    train_dataset_eog.save_to_disk("./dataset/EOG/train")
    test_dataset_eog.save_to_disk("./dataset/EOG/test")

    """==========================================mix EEG with EMG===================================================="""
    eeg, emg = nums_match(EEG, EMG)
    res_emg = generate_dataset(eeg, emg, dicts={"train": 0.8, "valid": 0.2})

    train_dataset_emg = Dataset.from_dict({"x": res_emg["train"]["x"],
                                           "y": res_emg["train"]["y"],
                                           "std": res_emg["train"]["std"],
                                           })
    test_dataset_emg = Dataset.from_dict({"x": res_emg["valid"]["x"],
                                          "y": res_emg["valid"]["y"],
                                          "std": res_emg["valid"]["std"],
                                          })
    train_dataset_emg.save_to_disk("./dataset/EMG/train")
    test_dataset_emg.save_to_disk("./dataset/EMG/test")


if __name__ == "__main__":
    # ===============================init dataset=============================
    init_dataset()

    # ================================load dataset======================
    # ===========================test use ===============================================
    # loaded_train_dataset = Dataset.load_from_disk("./dataset/EMG/train")
    # # loaded_train_dataset = Dataset.load_from_disk("./dataset/EOG/train")
    # # loaded_train_dataset.set_transform()
    # loaded_train_dataset.set_format("numpy")  # set format to numpy
    # print("Train Dataset - x:", loaded_train_dataset["x"][2])
    # print("y:", loaded_train_dataset["y"][2])
    # print("std:", loaded_train_dataset["std"][2])
    # show_img(x=loaded_train_dataset["y"][9], y=None, pre=None, save_name='./img-noise.svg',format='svg', dpi=50, bbox_inches='tight')
