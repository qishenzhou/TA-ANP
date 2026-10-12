"""TA-ANP model registration and lookup."""

from ._model_register import ModelRegistry
from .TAANP import TAANP


# Expose the registry and convenience functions
def get_model(name):
    """Get a model by name"""
    return ModelRegistry.get_model(name)


def list_models():
    """List all available models"""
    return ModelRegistry.list_models()


# Export main interfaces
__all__ = ['ModelRegistry', 'get_model', 'list_models']
