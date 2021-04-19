nn_cfg: Dict[str, List[int]] = {
    'dataset': ["exp", "ln", "permute", "iris"],
    'masking': ["chord", "no_chord"],
    'optimizer': ["Adam", "SGD", "RMRProp"],
    'LR': ["0.1", "0.01", "1e-3", "1e-4", "1e-5", "1e-6", "1e-7"],
}
