# Sparse Factorization of Large Square Matrices
# **Architecture**
![Architeture of PSF](https://github.com/RuslanKhalitov/SparseFactorization/blob/master/psf.pdf?raw=true)
PSF-Attn is a transformer-like model, where we replace the scaled dot-product attention with a product of sparse square matrices. The matrix product provides an approximation to a full non-normalized attention matrix.

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
To reproduce the synthetic data experiment. You should first generate data via [synth_data_generation.py](https://github.com/RuslanKhalitov/SparseFactorization/tree/master/SyntheticExperiments) for the two tasks, Adding and Temporal Order.  You can choose from the sequence length range `[2**7, 2**8, 2**9, 2**10, 2**11, 2**12, 2**13, 2**14]`. The generated dataset will be stored as tensors. 

To train PSF and X-formers, you can just run [psf)training.py](https://github.com/RuslanKhalitov/SparseFactorization/tree/master/SyntheticExperiments) and [xformer_training.py](https://github.com/RuslanKhalitov/SparseFactorization/tree/master/SyntheticExperiments). You can transfer to a different task by changing the following settings.

    cfg_model = config['order']['models']['Transformer']  
    cfg_training = config['order']['training']

We provide a default configuration for each model of each task in [synthetic_training_config.py](https://github.com/RuslanKhalitov/SparseFactorization/tree/master/SyntheticExperiments). For instance, we use the following setting for PSF on Adding problem.

    "PSF":{  
    "add_init_linear_layer": True,  
    "vocab_size": 1,  
    "dim": 32,  
    "Ws": [32, 'GELU'],  
    "V": [32, 'GELU'],  
    "pooling_type": "FLATTEN",  
    "head": ['linear'],  
    "n_class": 1,  
    "n_channels_V": 8,  
    "use_cuda": True,  
    "use_residuals": True,  
    "use_pos_embedding": False,  
    "problem": "adding"}

## Long Range Arena
Explain how to 
  1) Install the dataset
  2) Preprocess data
  3) Run each experiment
  4) Visualise the attention maps at test time

