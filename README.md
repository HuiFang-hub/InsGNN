# InsGNN
This repo contains the sample code for reproducing the results of our paper: **InsGNN: Interpretable Spatio-temporal Graph Neural Networks via Information Bottleneck**.
## Introduction
Spatio-temporal graph neural networks (STGNNs) have garnered considerable attention for their promising performance across various applications. While existing models have demonstrated superior performance in exploring the interpretability of graph  neural networks (GNNs), the interpretability of STGNNs is constrained by their complex spatio-temporal correlations. In this paper,  we introduce a novel approach named INterpretable Spatio-temporal Graph Neural Network (InsGNN), which aims to elucidate the  predictive process of STGNNs by **identifying key components**. 

## Quick Start
### 1. Requirements
Please first install python (we use 3.10.6) and then install other dependencies by
```bash
pip install -r requirements.txt
```
### 2. Run
```bash
bash scripts/{dataset}.bash
```
 such as 
 ```bash
 bash scripts/mutag.bash
 ```
or
 ```bash
 python main.py --cfg configs/GIN-spmotif_0.9.yaml  device 4  seed 0  framework InsGNN data.data_name spmotif_0.9 train.epochs 150  model.normal_coef 0.0  model.kl_1_coef 0.001  model.kl_2_coef 0.0  model.GC_delta_coef 0.0
 ```
