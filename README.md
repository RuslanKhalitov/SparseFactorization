# Sparse Factorization of Large Square Matrices

## Install requirements.txt
All the system requirements (used packages and their versions) are provided in requirements.txt.
The most important packages are torch-sparse and torch-geometric. The batched sparse multiplication described in the main paper is build upon torch_sparse.spmm and torch_geometric.DataLoader functions. More details are provided in <FILE.PY>

## Non-parametric experiments
To run the code you should:
1. Download the directory "**non-parametric"**;
2. Unzip **Square_matrices**;
3. run **sf_appr_test_all.m** in Matlab.

Explaination of the files we use in this part:
1. **chord_mask_mat.m** is used to generate chord masking for W;
2. **data_list.csv** is a list used to go though all the pictures;
3. **load_suqare_matrix.m** is a dataloader used to transform images to matrices;
4. **sf_appr_test.m** is the sparse approximation for one instance. It includes objective function, gradient update test process, and also TSVD part.
5. **sf_appr_test_all.m** is the sparse approximation for all instances.

## Large Attention Matrices
A step-by-step explanation of how to run synthetic experiments code.

## Long Range Arena
Explain how to 
  1) Install the dataset
  2) Preprocess data
  3) Run each experiment
  4) Visualise the attention maps at test time

