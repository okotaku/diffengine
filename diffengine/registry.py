from mmengine.registry import DATASETS as MMENGINE_DATASETS
from mmengine.registry import FUNCTIONS as MMENGINE_FUNCTIONS
from mmengine.registry import HOOKS as MMENGINE_HOOKS
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmengine.registry import Registry

__all__ = ['MODELS', 'DATASETS', 'HOOKS', 'FUNCTIONS']

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
