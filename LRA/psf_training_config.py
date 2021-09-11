config = {
    "listops":{
        "model":{
            "vocab_size":15 + 1 + 1, # 15 tokens + 1 PAD + 1 CLS
            "embedding_size": 512,
            "n_vec": 1999 + 1, # 1999 sequence length + 1 CLS
            "n_W": 11,
            "Ws": [128, 'GELU'],
            "V": [128, 'GELU'],
            "n_channels_V": 128,
            "n_class": 10,
            "pooling_type": "CLS", # "FLATTEN" or "CLS"
            "head": ['linear'], # ['linear'] or ['non-linear', 32], the second value is the number of hidden neurons
            "use_cuda": True,
            "use_residuals": False,
            "dropout1_p": 0, # Directly after the embeddings
            "dropout2_p": 0, # After V 
            "dropout3_p": 0, # Before the final layer
            "init_embedding_weights": False,
            "use_pos_embedding": True,
            "problem": "listops"
        },
        "training":{
            "device_id": 1,
            "batch_size":32,
            "learning_rate":0.001,
            "eval_frequency":1,
            "num_train_steps":7
        }
    },
    "cifar10":{
        "model":{
            "vocab_size": 256, # 256 unique pixel values
            "embedding_size": 16,
            "n_vec": 1024,
            "n_W": 10,
            "Ws": [16, 'GELU'],
            "V": [16, 'GELU'],
            "n_channels_V": 16,
            "n_class": 10,
            "pooling_type": "FLATTEN",
            "head": ['non-linear', 16],
            "use_cuda": True,
            "use_residuals": False,
            "dropout1_p": 0,
            "dropout2_p": 0.2,
            "dropout3_p": 0.8,
            "init_embedding_weights": False,
            "use_pos_embedding": True,
            "problem": "cifar10"
        },
        "training":{
            "device_id": 0,
            "batch_size":32,
            "learning_rate":0.001,
            "eval_frequency":1,
            "num_train_steps":35
        }
    },
    "pathfinder":{
        "model":{
            "vocab_size": 225, # 225 unique pixel values
            "embedding_size": 32,
            "n_vec": 1024,
            "n_W": 11,
            "Ws": [128, 'GELU'],
            "V": [128, 'GELU'],
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
        "training":{
            "device_id": 0,
            "batch_size": 64,
            "learning_rate":0.001,
            "eval_frequency":1,
            "num_train_steps": 66
        }
    },
    "imdb":{
        "model":{
            "vocab_size": 95 + 1 + 1, # 95 unique symbols + 1 PAD + 1 CLS
            "embedding_size": 32,
            "n_vec": 4096 + 1, # Plus CLS token
            "n_W": 12,
            "Ws": [128, 'GELU'],
            "V": [128, 'GELU'],
            "n_channels_V": 32,
            "n_class": 2,
            "pooling_type": "CLS", # "FLATTEN" or "CLS"
            "head": ['linear'], # ['linear'] or ['non-linear', 32], the second value is the number of hidden neurons
            "use_cuda": True,
            "use_residuals": True,
            "dropout1_p": 0.4,
            "dropout2_p": 0,
            "dropout3_p": 0,
            "init_embedding_weights": True,
            "use_pos_embedding": False,
            "problem": "imdb"
        },
        "training":{
            "device_id": 0,
            "batch_size":32,
            "learning_rate":0.0001,
            "eval_frequency":1,
            "num_train_steps":145
        }
    }
}

