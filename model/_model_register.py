from typing import Dict, Type

from torch import nn


class ModelRegistry:
    """Model registry for managing all available models"""

    _models: Dict[str, Type[nn.Module]] = {}

    @classmethod
    def register(cls, name: str = None):
        """
        Model registration decorator
        Args:
            name: Model name, if None, use class name
        """

        def wrapper(model_cls):
            model_name = name or model_cls.__name__
            if model_name in cls._models:
                raise ValueError(f"Model {model_name} already registered!")
            cls._models[model_name] = model_cls
            return model_cls

        return wrapper

    @classmethod
    def get_model(cls, name: str) -> Type[nn.Module]:
        """
        Get registered model class
        Args:
            name: Model name
        Returns:
            Model class
        Raises:
            ValueError: If model not found
        """
        if name not in cls._models:
            raise ValueError(
                f"Model {name} not found! Available models: {list(cls._models.keys())}")
        return cls._models[name]

    @classmethod
    def list_models(cls) -> list:
        """List all registered models"""
        return list(cls._models.keys())
