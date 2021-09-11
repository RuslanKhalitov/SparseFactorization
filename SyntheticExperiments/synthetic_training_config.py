config = {
    "adding":{
        "models":{
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
                "problem": "adding"
            },
            "Transformer":{
                "name": "transformer",
                "vocab_size": 1,
                "add_init_linear_layer": True,
                "dim": 32,
                "depth": 1,
                "heads": 1,
                "pooling_type": "FLATTEN",
                "head": ['linear'],
                "n_class": 1,
                "use_cuda": True,
                "use_pos_embedding": False,
                "problem": "adding"
            },
            "Linformer":{
                "name": "linformer",
                "vocab_size": 1,
                "add_init_linear_layer": True,
                "dim": 32,
                "depth": 1,
                "heads": 1,
                "pooling_type": "FLATTEN",
                "head": ['linear'],
                "n_class": 1,
                "use_cuda": True,
                "use_pos_embedding": False,
                "problem": "adding"
            },
            "Performer":{
                "name": "performer",
                "vocab_size": 1,
                "add_init_linear_layer": True,
                "dim": 32,
                "depth": 1,
                "heads": 1,
                "pooling_type": "FLATTEN",
                "head": ['linear'],
                "n_class": 1,
                "use_cuda": True,
                "use_pos_embedding": False,
                "problem": "adding"
            },
        },
        "training":{
            "device_id": 0,
            "batch_size":40,
            "learning_rate":0.001,
            "eval_frequency":1,
            "num_train_steps":20 # Fixed for all models
        }
    },
    "order":{
        "models":{
            "PSF":{
                "add_init_linear_layer": True,
                "dim": 32,
                "Ws": [32, 'GELU'],
                "V": [32, 'GELU'],
                "pooling_type": "FLATTEN",
                "head": ['linear'],
                "n_class": 4,
                "n_channels_V": 8,
                "use_cuda": True,
                "use_residuals": True,
                "use_pos_embedding": True,
                "problem": "order"
            },
            "Transformer":{
                "name": "transformer",
                "vocab_size": 6,
                "add_init_linear_layer": True,
                "dim": 32,
                "depth": 1,
                "heads": 1,
                "pooling_type": "FLATTEN",
                "head": ['linear'],
                "n_class": 4,
                "use_cuda": True,
                "use_pos_embedding": True,
                "problem": "order"
            },
            "Linformer":{
                "name": "linformer",
                "vocab_size": 6,
                "add_init_linear_layer": True,
                "dim": 32,
                "depth": 1,
                "heads": 1,
                "pooling_type": "FLATTEN",
                "head": ['linear'],
                "n_class": 4,
                "use_cuda": True,
                "use_pos_embedding": True,
                "problem": "order"
            },
            "Performer":{
                "name": "performer",
                "vocab_size": 6,
                "add_init_linear_layer": True,
                "dim": 32,
                "depth": 1,
                "heads": 1,
                "pooling_type": "FLATTEN",
                "head": ['linear'],
                "n_class": 4,
                "use_cuda": True,
                "use_pos_embedding": True,
                "problem": "order"
            },
        },
        "training":{
            "device_id": 0,
            "batch_size": 40,
            "learning_rate": 0.001,
            "eval_frequency": 1,
            "num_train_steps": 20 # Fixed for all models
        }
    }
}
