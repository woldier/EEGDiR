## EEGDiR: Electroencephalogram denoising network for temporalinformation storage and global modeling through Retentive Network

<b>
Bin Wang, 
<a href='https://dengfei-ailab.github.io'>Deng Fei</a>, 
<a href='https://github.com/jiangpeifan'>Peifan Jiang</a>
</b>

<hr>
<i>Electroencephalogram (EEG) signals play a pivotal role in clinical medicine, brain research, and
neurological disease studies. However, susceptibility to various physiological and environmental artifacts introduces noise in recorded EEG data, impeding accurate analysis of underlying brain activity.
Denoising techniques are crucial to mitigate this challenge. Recent advancements in deep learningbased approaches exhibit substantial potential for enhancing the signal-to-noise ratio of EEG data
compared to traditional methods. In the realm of large-scale language models (LLMs), the Retentive
Network (Retnet) infrastructure, prevalent for some models, demonstrates robust feature extraction
and global modeling capabilities. Recognizing the temporal similarities between EEG signals and
natural language, we introduce the Retnet from natural language processing to EEG denoising. This
integration presents a novel approach to EEG denoising, opening avenues for a profound understanding
of brain activities and accurate diagnosis of neurological diseases. Nonetheless, direct application
of Retnet to EEG denoising is unfeasible due to the one-dimensional nature of EEG signals, while
natural language processing deals with two-dimensional data. To facilitate Retnet application to EEG
denoising, we propose the signal embedding method, transforming one-dimensional EEG signals into
two dimensions for use as network inputs. Experimental results validate the substantial improvement
in denoising effectiveness achieved by the proposed method.</i>

## Package dependencies
The project is built with PyTorch 1.13.1, Python3.10, CUDA11.7. For package dependencies, you can install them by:
```bash
pip install -r requirements.txt
```
![Uformer](image/fig2.jpg)
## How to get dataset:
- dataset is available at (coming soon): https://huggingface.co/datasets/woldier/EEGDiRDataset
