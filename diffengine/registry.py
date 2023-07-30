from mmengine.registry import DATASETS as MMENGINE_DATASETS
from mmengine.registry import FUNCTIONS as MMENGINE_FUNCTIONS
from mmengine.registry import HOOKS as MMENGINE_HOOKS
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmengine.registry import TRANSFORMS as MMENGINE_TRANSFORMS
from mmengine.registry import Registry

__all__ = ['MODELS', 'DATASETS', 'HOOKS', 'FUNCTIONS', 'TRANSFORMS']

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
