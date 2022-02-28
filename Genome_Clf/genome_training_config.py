config = {
    "DDcDNA":{
        "PSF":{
            "name": "psf",
            "vocab_size":6,
            "embedding_size": 32,
            "n_vec": 16384,
            "n_W": 14,
            "Ws": [32, 'GELU'],
            "V": [32, 'GELU'],
            "n_channels_V": 32,
            "n_class": 2,
            "pooling_type": "FLATTEN", # "FLATTEN" or "CLS"
            "head": ['linear'], # ['linear'] or ['non-linear', 32], the second value is the number of hidden neurons
            "use_cuda": True,
            "use_residuals": False,
            "dropout1_p": 0.2, # Directly after the embeddings
            "dropout2_p": 0, # After V
            "dropout3_p": 0.8, # Before the final layer
            "init_embedding_weights": False,
            "use_pos_embedding": False,
        },
        "Linformer":{
            "name": "linformer",
            "vocab_size": 6,
            "max_seq_len": 16384,
            "add_init_linear_layer": False,
            "dim": 32,
            "depth": 4,
            "heads": 2,
            "pooling_type": "FLATTEN",
            "head": ['linear'],
            "n_class": 2,
            "use_cuda": True,
            "pos_embedding": ['APC'],
            "problem": "encode"
        },
        "Performer":{
            "name": "performer",
            "vocab_size": 6,
            "max_seq_len": 16384,
            "add_init_linear_layer": False,
            "dim": 32,
            "depth": 1,
            "heads": 1,
            "pooling_type": "FLATTEN",
            "head": ['linear'],
            "n_class": 2,
            "use_cuda": True,
            "pos_embedding": ['APC'],
            "problem": "encode"
        },
        "Transformer":{
            "name": "transformer",
            "vocab_size": 6,
            "max_seq_len": 16384,
            "add_init_linear_layer": False,
            "dim": 32,
            "depth": 4,
            "heads": 4,
            "pooling_type": "FLATTEN",
            "head": ['linear'],
            "n_class": 2,
            "use_cuda": True,
            "pos_embedding": ['APC'],
            "problem": "encode"
        },
        "Nystromformer":{
            "name": "nystromformer",
            "vocab_size": 6,
            "max_seq_len": 16384,
            "add_init_linear_layer": False,
            "dim": 32,
            "depth": 1,
            "heads": 1,
            "pooling_type": "FLATTEN",
            "head": ['linear'],
            "n_class": 2,
            "use_cuda": True,
            "pos_embedding": ['APC'],
            "problem": "encode"
        },
        "LStransformer": {
            "name": "lstransformer",
            "vocab_size": 6,
            "max_seq_len": 16384,
            "add_init_linear_layer": False,
            "dim": 32,
            "depth": 1,
            "heads": 1,
            "pooling_type": "FLATTEN",
            "head": ['linear'],
            "n_class": 2,
            "use_cuda": True,
            "pos_embedding": ['APC'],
            "problem": "encode"
        },
        "Reformer": {
            "name": "reformer",
            "vocab_size": 6,
            "max_seq_len": 16384,
            "add_init_linear_layer": False,
            "dim": 32,
            "depth": 1,
            "heads": 1,
            "pooling_type": "FLATTEN",
            "head": ['linear'],
            "n_class": 2,
            "use_cuda": True,
            "pos_embedding": ['APC'],
            "problem": "encode"
        },
        "training":{
            "device_id": 0,
            "batch_size":16,
            "learning_rate":0.0001,
            "eval_frequency":1,
            "num_train_steps":100
        },
    }
}

