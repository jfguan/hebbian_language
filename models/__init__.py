def build_model(cfg):
    model_type = cfg.pop("model")
    if model_type == "hebbian_mamba":
        from models.hebbian_mamba import Config, HebbianMamba
        model_cfg = Config(**{k: v for k, v in cfg.items() if hasattr(Config, k)})
        return HebbianMamba(model_cfg), model_cfg, "HebbianMamba"
    elif model_type == "mamba":
        from models.mamba import Config, Mamba
        model_cfg = Config(**{k: v for k, v in cfg.items() if hasattr(Config, k)})
        return Mamba(model_cfg), model_cfg, "Mamba"
    elif model_type == "hebbian_minimal":
        from models.hebbian_minimal import Config, HebbianConv
        model_cfg = Config(**{k: v for k, v in cfg.items() if hasattr(Config, k)})
        return HebbianConv(model_cfg), model_cfg, "HebbianConv"
    else:
        raise ValueError(f"Unknown model type: {model_type!r}")
