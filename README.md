## Non-parametric experiments
To run the code you should:
1. Download the directory **non-parametric**;
2. Unzip [Square_matrices];
3. run [sf_appr_test_all.m]in Matlab.

Explaination of the files we use in this part:
1. ***chord_mask_mat.m*** is used to generate chord masking for W;
2. ***datalist.csv*** is a list used to go though all the pictures;
3. ***load_square_matrix.m*** is a dataloader used to transform images to matrices;
4. ***sf_appr_test.m*** is the sparse approximation for one instance. It includes objective function, gradient update test process, and also TSVD part.
5. ***sf_appr_test_all.m*** is the sparse approximation for all instances.

## Large Attention Matrices
To reproduce the synthetic data experiment. You should first generate data via ***synth_data_generation.py*** for the two tasks, Adding and Temporal Order.  You can choose from the sequence length range `[2**7, 2**8, 2**9, 2**10, 2**11, 2**12, 2**13, 2**14]`. The generated dataset will be stored as tensors. 

To train PSF and X-formers, you can just run ***psf_training.py*** and ***xformer_training.py***. You can transfer to a different task by changing the following settings.

    cfg_model = config['order']['models']['Transformer']  
    cfg_training = config['order']['training']

We provide a default configuration for each model of each task in ***synthetic_training_config.py***. For instance, we use the following setting for PSF on Adding problem.
