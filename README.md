# CDGCL: Curriculum Diffusion-based Graph Contrastive learning
We provide the code (in pytorch) and datasets for our paper: "Curriculum Diffusion-based Graph Contrastive learning" (CDGCL)

## 1. Desription
The repository is organised as follows:

* data/: contains the 4 benchmark datasets: Cora, Citeseer, Coauthor-CS and Actor. All datasets will be processed on the fly. Please extract the compressed file of each dataset before running.

* model/: contains our models.


## 2. Requirements
To install required packages
- pip install -r requirements.txt


## 3. Experiments
For reproducibility, please run these commands regarding to specific dataset:

- python main_cond.py --dataset=Cora/Citeseer/CS/Actor

## 4. Citation