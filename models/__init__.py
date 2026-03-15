from dataclasses import asdict


def build_model(mc):
    fields = asdict(mc)
    model_type = fields.pop("model")
    if model_type == "hebbian_mamba":
        from .hebbian_mamba import Config, HebbianMamba
        model_cfg = Config(**{k: v for k, v in fields.items() if hasattr(Config, k) and v is not None})
        return HebbianMamba(model_cfg), model_cfg, "HebbianMamba"
    elif model_type == "mamba":
        from .mamba import Config, Mamba
        model_cfg = Config(**{k: v for k, v in fields.items() if hasattr(Config, k) and v is not None})
        return Mamba(model_cfg), model_cfg, "Mamba"
    elif model_type == "hebbian_minimal":
        from .hebbian_minimal import Config, HebbianConv
        model_cfg = Config(**{k: v for k, v in fields.items() if hasattr(Config, k) and v is not None})
        return HebbianConv(model_cfg), model_cfg, "HebbianConv"
    else:
        raise ValueError(f"Unknown model type: {model_type!r}")
