from mmengine.registry import DATA_SAMPLERS as MMENGINE_DATA_SAMPLERS
from mmengine.registry import DATASETS as MMENGINE_DATASETS
from mmengine.registry import FUNCTIONS as MMENGINE_FUNCTIONS
from mmengine.registry import HOOKS as MMENGINE_HOOKS
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmengine.registry import OPTIMIZERS as MMENGINE_OPTIMIZERS
from mmengine.registry import TRANSFORMS as MMENGINE_TRANSFORMS
from mmengine.registry import Registry

__all__ = [
    'MODELS', 'DATASETS', 'HOOKS', 'FUNCTIONS', 'TRANSFORMS', 'DATA_SAMPLERS',
    'OPTIMIZERS'
]

DATA_SAMPLERS = Registry(
    'data sampler',
    parent=MMENGINE_DATA_SAMPLERS,
    locations=['diffengine.datasets.samplers'])
DATASETS = Registry(
    'dataset',
    parent=MMENGINE_DATASETS,
    locations=['diffengine.datasets'],
)
MODELS = Registry(
    'model',
    parent=MMENGINE_MODELS,
    locations=['diffengine.models'],
)
OPTIMIZERS = Registry(
    'optimizer',
    parent=MMENGINE_OPTIMIZERS,
    locations=['diffengine.engine'],
)
HOOKS = Registry(
    'hook',
    parent=MMENGINE_HOOKS,
    locations=['diffengine.engine'],
)
FUNCTIONS = Registry(
    'function',
    parent=MMENGINE_FUNCTIONS,
    locations=['diffengine.datasets'],
)
TRANSFORMS = Registry(
    'transform',
    parent=MMENGINE_TRANSFORMS,
    locations=['diffengine.datasets.transforms'],
)
