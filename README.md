<div align="center">
<h1>MineNetCD </h1>
<h3>MaskCD: A Remote Sensing Change Detection Network Based on Mask Classification</h3>

[Weikang Yu](https://ericyu97.github.io/)<sup>1,2</sup>, [Xiaokang Zhang](https://xkzhang.info/)<sup>3</sup>, [Richard Gloaguen](https://scholar.google.de/citations?user=e1QDLQUAAAAJ&hl=de)<sup>2</sup>, [Xiao Xiang Zhu](https://www.asg.ed.tum.de/sipeo/home/)<sup>1</sup>, [Pedram Ghamisi](https://www.ai4rs.com/)<sup>2,4</sup>

<sup>1</sup> Technical University of Munich, <sup>2</sup> Helmholtz-Zentrum Dresden-Rossendorf (HZDR), <sup>3</sup> Wuhan University of Science and Technology, <sup>4</sup> Lancaster University

Paper: [IEEE TGRS 2024](https://ieeexplore.ieee.org/document/10744421) (DOI: 10.1109/TGRS.2024.3491715)
</div>



## Updates
``Nov. 9, 2024`` Our paper has been accepted on IEEE TGRS, and the code is released.
## Abstract
Monitoring land changes triggered by mining activities is crucial for industrial control, environmental management and regulatory compliance, yet it poses significant challenges due to the vast and often remote locations of mining sites. Remote sensing technologies have increasingly become indispensable to detect and analyze these changes over time. We thus introduce MineNetCD, a comprehensive benchmark designed for global mining change detection using remote sensing imagery. The benchmark comprises three key contributions. First, we establish a global mining change detection dataset featuring more than 70k paired patches of bitemporal high-resolution remote sensing images and pixel-level annotations from 100 mining sites worldwide. Second, we develop a novel baseline model based on a change-aware Fast Fourier Transform (ChangeFFT) module, which enhances various backbones by leveraging essential spectrum components within features in the frequency domain and capturing the channel-wise correlation of bitemporal feature differences to learn change-aware representations. Third, we construct a unified change detection (UCD) framework that currently integrates 20 change detection methods. This framework is designed for streamlined and efficient processing, utilizing the cloud platform hosted by HuggingFace. Extensive experiments have been conducted to demonstrate the superiority of the proposed baseline model compared with 19 state-of-the-art change detection approaches. Empirical studies on modularized backbones comprehensively confirm the efficacy of different representation learners on change detection. This benchmark represents significant advancements in the field of remote sensing and change detection, providing a robust resource for future research and applications in global mining monitoring. Dataset and Codes are available via the link.
## Overview
* **MaskCD** is a pioneering work introducing a mining change detection benchmark, including a global-scale mining change detection dataset, a ChangeFFT-based model, and a unified change detection framework.
<p align="center">
  <img src="figures/MineNetCDIntro.pdf" alt="architecture" width="80%">
</p>

* **Hierarchical transformer-based Siamese encoder** uses the window-shifted self-attention mechanism to simultaneously extract bitemporal deep features from remote sensing images.
<p align="center">
  <img src="figures/encoder.png" alt="architecture" width="80%">
</p>

* **Cross-Level Change Representation Perceiver** integrates deformable multi-head self-attention mechanism and an FPN to obtain multi-scale binary masks.
<p align="center">
  <img src="figures/clcrp.png" alt="architecture" width="80%">
</p>

* **Masked Cross-attention-based Decoder and Mask Classification module** processes query embeddings to obtain per-segment embeddings as foundations for generating mask embeddings and the class labels for the masks. 
<p align="center">
  <img src="figures/maskclassification.png" alt="architecture" width="80%">
</p>

## Getting started
### Environment Preparation
Create a conda environment for MaskCD
 ```console
conda create -n minenetcd
conda activate minenetcd
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers
pip install accelerate
pip install datasets

git clone https://github.com/MzeroMiko/VMamba.git
cd VMamba
pip install -r requirements.txt
cd kernels/selective_scan && pip install .

pip install scipy
```
Configurate the accelerate package:
```console
accelerate config
```
___
### Run the Experiments
#### Training a model:
```console
accelerate launch train.py HZDR-FWGEL/MineNetCD256 --batch-size 32 --learning-rate 5e-5 --epochs 100
```
``DATASET_ID`` is the repo_id of the dataset on Huggingface Hub. 

Avalaible examples used in MineNetCD: 

``ericyu/CLCD_Cropped_256`` 
``ericyu/LEVIRCD_Cropped_256`` 
``ericyu/SYSU_CD``
``ericyu/GVLM_Cropped_256``
``ericyu/EGY_BCD``

The model will be automatically saved under the path "./exp/``DATASET_ID``/", the model with the highest F1 score will be saved under "./exp/``DATASET_ID``/best_f1"
___
Testing a model:
```console
accelerate launch test.py --dataset HZDR-FWGEL/MineNetCD256 --model $MODEL_ID$
```
The ``MODEL_ID`` can be the path of your trained model (e.g., exp/``DATASET_ID``/best_f1)

___
Reproducing our results:

We have uploaded our pretrained model weights to the Huggingface Hub, the ``MODEL_ID`` is as follows:

``ericyu/MaskCD_CLCD_Cropped_256``
``ericyu/MaskCD_LEVIRCD_Cropped256``
``ericyu/MaskCD_SYSU_CD``
``ericyu/MaskCD_GVLM_Cropped_256``
``ericyu/MaskCD_EGY_BCD``

Here is an example pf reproducing the results of MaskCD on CLCD results:
```console
accelerate launch test.py --dataset ericyu/CLCD_Cropped_256 --model ericyu/MaskCD_CLCD_Cropped_256
```
___
Upload your model to Huggingface Hub

You can also push your model to Huggingface Hub by uncommenting and modifying the codeline in the ``test.py``:
```python
if accelerator.is_local_main_process:
    model = model.push_to_hub('ericyu/MaskCD_EGY_BCD')
```
___
Create your own dataset:

Please modify the ``dataset_creator.py`` and use ``save_to_disk`` or ``push_to_hub`` according to your usage.

More datasets/pre-trained models will be implemented to be available in our new ``UCD`` project, please stay tuned and star our ``UCD`` [Repo](https://github.com/EricYu97/UCD).

___


If you find MaskCD useful for your study, please kindly cite us:
```
@ARTICLE{10587034,
  author={Yu, Weikang and Zhang, Xiaokang and Das, Samiran and Zhu, Xiao Xiang and Ghamisi, Pedram},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={MaskCD: A Remote Sensing Change Detection Network Based on Mask Classification}, 
  year={2024},
  volume={62},
  number={},
  pages={1-16},
  keywords={Transformers;Feature extraction;Image segmentation;Decoding;Task analysis;Representation learning;Object oriented modeling;Change detection (CD);deep learning;deformable attention;mask classification (MaskCls);masked cross-attention;remote sensing (RS)},
  doi={10.1109/TGRS.2024.3424300}}
```

## Future Development Schedule:

We are developing a unified change detection (UCD) framework that implements more than 18 change detection approaches and have more than 70 available models. The codes will be released [here](https://github.com/EricYu97/UCD).

## Tutorial Avaiable!
We just added a very simple example as a tutorial for those who are interested in change detection, check [here](https://github.com/EricYu97/CDTutorial) for more details.

## Acknowledgement:

This codebase is heavily borrowed from [Transformers](https://github.com/huggingface/transformers) package.


