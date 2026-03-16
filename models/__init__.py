from train.configs import ModelConfig, ModelType


def build_model(model_config: ModelConfig):
    if model_config.model == ModelType.HEBBIAN:
        from .hebbian import HebbianConv
        return HebbianConv(model_config)
    elif model_config.model == ModelType.DELTA_HEBBIAN:
        from .delta_hebbian import DeltaHebbianConv
        return DeltaHebbianConv(model_config)
    elif model_config.model == ModelType.MAMBA:
        from .mamba import Mamba
        return Mamba(model_config)
    elif model_config.model == ModelType.GDN:
        from .gated_deltanet import GatedDeltaNet
        return GatedDeltaNet(model_config)
    else:
        raise ValueError(f"Unknown model type: {model_config.model!r}")
