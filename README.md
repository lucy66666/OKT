# OKT: Open-ended Knowledge Tracing

**OKT** provides the first exploration into open-ended knowledge tracing by studying the new task of predicting students’ exact open-ended responses to questions. This repository contains code for [Open-Ended Knowledge Tracing for Computer Science Education](https://arxiv.org/abs/2203.03716)\
Naiming Liu, Zichao Wang, Richard G. Baraniuk, Andrew Lan, to be presented at [EMNLP 2022](https://2022.emnlp.org/).

A block diagram of OKT is shown here:
<p align="center">
<img src="OKT-code.png" alt="Image" width="400"/>
</p>

## Dependencies
- python 3.8.12 
- torch 1.10.0 
- transformers 4.6.1 
- scikit-learn 0.24.2 
- numpy 1.22.1 
- munch 2.5.0 
- nltk 3.7
- neptune-client 0.14.2 

## Data
We use [CSEDM](https://sites.google.com/ncsu.edu/csedm-dc-2021/) dataset and preprocess the data by 1). removing all codes that can't be parsed as abstract syntax tree. 2). convert student codes to vector representation using [ASTNN](https://github.com/zhangj111/astnn). You can download the preprocessed data with the commands below.
### Download preprocessed data
```
cd scripts
bash data.sh
```

## Training OKT
In order to train OKT model, run `python main_okt.py` on the command line. All parameters can be changed in the `configs_okt.yaml` file. We use [Neptune.ai](https://neptune.ai/) to track our experiment results. If you also want to use Neptune.ai, you should change `neptune_project` and `neptune_api` in the parameter list to your own neptune credentials. \
**Note**: To use other knowledge tracing (KT) models instead of LSTM as knowledge estimation for OKT, you should use pre-trained KT models. We integrate two KT models (AKT, DKVMN) in our code (need to uncomment first). If you want to use them, please follow [AKT](https://github.com/arghosh/AKT) and [DKVMN](https://github.com/jennyzhang0215/DKVMN) repo to pretrain corresponding KT models. 

## Pre-trained models
### Download pre-trained GPT models
We provide two fine-tuned GPT-2 models to test the effect of pre-trained response generation model. One with [funcom](https://arxiv.org/pdf/1904.02660v1.pdf) dataset, while the other is further on [CSEDM](https://sites.google.com/ncsu.edu/csedm-dc-2021/) based on the first one. Models can be downloaded with the following commands.
```
cd scripts
bash pretrained_lm.sh
```

### Training LSTM and classifier
In order to pre-train knowledge estimation (LSTM) and classifier, run `python main_student_model` on the command line. All parameters can be changed in the `configs_student_model.yaml` file.

## Results and Evaluation
We use two metrics: [CodeBLEU](https://github.com/microsoft/CodeXGLUE/blob/main/Code-Code/code-to-code-trans/CodeBLEU.MD) and [Dist-N](https://aclanthology.org/N16-1014.pdf) and integrate their codes into this repo. To understand more about evalution metrics, please follow their corresponding websites. Training models and generation results will be saved in a directory `checkpoints\$TIME` you just created, where `$TIME` is the current time in `data_time` format. It will contain two models (`lstm` for knowledge tracing and `model` for generative model). It also includes an `eval_log.pkl` file, which shows CodeBLEU score, Dist-1 and generated student answers together with ground-truth answers for comparison. A set of trained results can be downloaded here. 
```
cd scripts
bash results.sh
```

## Citations
Please cite our paper if your find it helpful to you work! 
```bibtex
@article{liu2022open,
  title={Open-Ended Knowledge Tracing},
  author={Liu, Naiming and Wang, Zichao and Baraniuk, Richard G and Lan, Andrew},
  journal={arXiv preprint arXiv:2203.03716},
  year={2022}
}
