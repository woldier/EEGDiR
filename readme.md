## EEGDiR: Electroencephalogram denoising network for temporalinformation storage and global modeling through Retentive Network

<b>
Bin Wang, 
<a href='https://dengfei-ailab.github.io'>Deng Fei</a>, 
<a href='https://github.com/jiangpeifan'>Peifan Jiang</a>
</b>

<hr>
<i>Electroencephalogram (EEG) signals are essential in clinical medicine, brain research, and neurological disease studies. However, various physiological and environmental artifacts introduce noise, 
hindering accurate analysis of brain activity. Recent advancements in deep learning offer promising solutions for enhancing EEG signal denoising compared to traditional methods. In this study, 
we introduce the Retentive Network architecture from large language model (LLM) to EEG signal denoising, leveraging its robust feature extraction and global modeling capabilities. 
To adapt Retnet to the one-dimensional nature of EEG signals, we propose a signal embedding method, 
transforming EEG signals into two dimensions for network input. This integration presents a novel approach 
to EEG denoising, opening avenues for a profound understanding of brain activities and accurate diagnosis 
of neurological diseases. In addition, since the creation of deep learning datasets is very time-consuming 
and cumbersome, we also provide a standardized ready-to-use dataset that has been preprocessed to accelerate 
the progress of deep learning methods. Based on the dataset we produced and the proposed method, t
he experimental results show that the denoising effect is significantly improved compared to the existing methods.</i>



---
![EEGDiR](image/fig2.jpg)
## How to get dataset:
dataset is available at: https://huggingface.co/datasets/woldier/eeg_denoise_dataset

The method chosen in this article is to download and extract `*.tar.gz` to the `{your_path}/data` directory.

The structure is as follows:
```text
data/dataset/
├── EMG
    ├── train
    ├── test
├── EOG
    ├── train
    ├── test
├── SS-EOG
    ├── train
    ├── test
└── test
```

---

## Package dependencies
The project is built with `PyTorch 1.13.1`, `Python3.10`, `CUDA11.7`. For package dependencies, you can install them by:
```bash
pip install -r requirements.txt
```

## How to run
1. Single GPU pycharm run
The easiest way is to run `trian.py` directly under pycham

This library uses the Accelerate library, so you don't need to make any modifications to this training framework. Accelerate will automatically convert your data to the GPU (if supported).
2. Multi-GPU run in CMD

If you want to train on multiple GPUs, you can refer to the following code

Before running it for the first time, you need to generate a configuration file through accelerate, telling accelerate which GPUs are currently involved in the training.
```shell
accelerate config
```
![accelerate_config](image/accelerate_config.png)

Once you have configured the accelerate config file, you can view the configuration via `accelerate env`(Optional).

Multi-GPU training can then be performed by calling `train.py`.
```shell
accelerate launch --gpu_ids=all  {your_path}/train.py
```
For more details, please refer to the Accelerate website: 
https://huggingface.co/docs/accelerate/basic_tutorials/launch
3. Multi-GPU run in pycharm 

If you find the cmd call scripts inelegant, you can also configure pycharm.

- Create a New Run/Debug Configuration
- Setting the ENV of RUN
- Set the mode to module, and give the model name: `accelerate.commands.launch`.
- Set the parameter `--gpu_ids=all` to set the available GPUs, and specify the script path `{your_path}/trian.py`.

![pycharm_config](image/pycharm_config.png)

enjoy!

---
This training framework is not perfect, 
I will continue to update it. 

If you have any ideas on how to improve this framework please issue or PR.
