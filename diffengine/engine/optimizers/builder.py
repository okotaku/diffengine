from transformers import Adafactor

from diffengine.registry import OPTIMIZERS


def register_transformer_optimizers():
    transformer_optimizers = []
    OPTIMIZERS.register_module(name='Adafactor')(Adafactor)
    transformer_optimizers.append('Adafactor')
    return transformer_optimizers


TRANSFORMER_OPTIMIZERS = register_transformer_optimizers()
