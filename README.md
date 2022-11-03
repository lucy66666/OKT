# OKT

**OKT** provides the first exploration into open-ended knowledge tracing by studying the new task of predicting students’ exact open-ended responses to questions. This repository contains code for [Open-Ended Knowledge Tracing for Computer Science Education](https://arxiv.org/abs/2203.03716), to be presented at EMNLP 2022.

A block diagram of OKT is shown here:
<p align="center">
<img src="OKT-code.png" alt="Image" width="400"/>
</p>

### Dependencies

### Training OKT
In order to train OKT model, run `python main_okt.py` on the command line. All parameters can be changed in the `configs_okt.yaml` file. We use [Neptune.ai](https://neptune.ai/) to track our experiment results. If you also want to use Neptune.ai, you should change `neptune_project` and `neptune_api` in the parameter list to your own neptune credentials. 

### Evaluation and Results
We use two metrics: CodeBLEU and Dist-1. (TODO: need finish)
Training models and generation results will be saved in a directory `checkpoints\$TIME` you just created, where `$TIME` is the current time in `data_time` format. It will contain two models (`lstm` for knowledge tracing and `model` for generative model). It also includes an `eval_log.pkl` file, which shows CodeBLEU score, Dist-1 and generated student answers together with ground-truth answers for comparison.

### Citations
Please cite our paper if your find it helpful to you work! 
```bibtex
@article{liu2022open,
  title={Open-Ended Knowledge Tracing},
  author={Liu, Naiming and Wang, Zichao and Baraniuk, Richard G and Lan, Andrew},
  journal={arXiv preprint arXiv:2203.03716},
  year={2022}
}
