.. role:: hidden
    :class: hidden-section

diffengine.models
===================================

.. contents:: diffengine.models
   :depth: 2
   :local:
   :backlinks: top

.. currentmodule:: diffengine.models

Editors
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   StableDiffusion
   StableDiffusionControlNet
   StableDiffusionXL
   StableDiffusionXLControlNet

Data Preprocessors
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   SDDataPreprocessor
   SDControlNetDataPreprocessor
   SDXLDataPreprocessor
   SDXLControlNetDataPreprocessor

Losses
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   L2Loss
   SNRL2Loss

diffengine.datasets
===================================

.. contents:: diffengine.datasets
   :depth: 2
   :local:
   :backlinks: top

.. currentmodule:: diffengine.datasets

Datasets
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   HFDataset
   HFControlNetDataset
   HFDreamBoothDataset

Transforms
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   PackInputs
   SaveImageShape
   RandomCrop
   CenterCrop
   RandomHorizontalFlip
   ComputeTimeIds
   DumpImage

diffengine.engine
===================================

.. contents:: diffengine.engine
   :depth: 2
   :local:
   :backlinks: top

.. currentmodule:: diffengine.engine

Hooks
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   VisualizationHook
   LoRASaveHook
   SDCheckpointHook
   UnetEMAHook
   VisualizationHook
