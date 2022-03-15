config = {
    "pathfinder":{
        "models":{
            "PSF":{
                "vocab_size": 225, # 225 unique pixel values
                "embedding_size": 32,
                "n_vec": 1024,
                "n_W": 11,
                "Ws": [64, 'GELU'],
                "V": [64, 'GELU'],
                "n_channels_V": 32,
                "n_class": 2,
                "pooling_type": "FLATTEN", # "FLATTEN" or "CLS"
                "head": ['linear'], # ['linear'] or ['non-linear', 32], the second value is the number of hidden neurons
                "use_cuda": True,
                "use_residuals": False,
                "dropout1_p": 0,
                "dropout2_p": 0,
                "dropout3_p": 0,
                "init_embedding_weights": False,
                "use_pos_embedding": True,
                "problem": "pathfinder"
            },
            "Transformer":{
                "name": "transformer",
                "vocab_size": 225,
                "add_init_linear_layer": False,
                "dim": 32,
                "depth": 1,
                "heads": 1,
                "pooling_type": "FLATTEN",
                "head": ['linear'],
                "n_class": 2,
                "use_cuda": True,
                "use_pos_embedding": True,
                "problem": "pathfinder"
            },
            "Linformer":{
                "name": "linformer",
                "vocab_size": 225,
                "add_init_linear_layer": False,
                "dim": 32,
                "depth": 1,
                "heads": 1,
                "pooling_type": "FLATTEN",
                "head": ['linear'],
                "n_class": 2,
                "use_cuda": True,
                "use_pos_embedding": True,
                "problem": "pathfinder"
            },
            "Performer":{
                "name": "performer",
                "vocab_size": 225,
                "add_init_linear_layer": True,
                "dim": 32,
                "depth": 1,
                "heads": 1,
                "pooling_type": "FLATTEN",
                "head": ['linear'],
                "n_class": 2,
                "use_cuda": True,
                "use_pos_embedding": True,
                "problem": "pathfinder"
            },
        },
        "training":{
            "device_id": 1,
            "batch_size":40,
            "learning_rate":0.001,
            "eval_frequency":1,
            "num_train_steps":20 # Fixed for all models 
        }
    }

}
