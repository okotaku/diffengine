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

   DeepFloydIF
   DistillSDXL
   ESDXL
   IPAdapterXL
   IPAdapterXLPlus
   StableDiffusion
   StableDiffusionControlNet
   StableDiffusionXL
   StableDiffusionXLControlNet
   StableDiffusionXLT2IAdapter

Data Preprocessors
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   ESDXLDataPreprocessor
   IPAdapterXLDataPreprocessor
   SDDataPreprocessor
   SDControlNetDataPreprocessor
   SDXLDataPreprocessor
   SDXLControlNetDataPreprocessor

Pipelines
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   IPAdapterXLPipeline
   IPAdapterXLPlusPipeline

Losses
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   DeBiasEstimationLoss
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
   HFDatasetPreComputeEmbs
   HFControlNetDataset
   HFDreamBoothDataset
   HFESDDatasetPreComputeEmbs

diffengine.datasets.transforms
===================================

.. contents:: diffengine.datasets.transforms
   :depth: 2
   :local:
   :backlinks: top

.. currentmodule:: diffengine.datasets.transforms

Transforms
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   BaseTransform
   DumpImage
   PackInputs
   RandomCrop
   CenterCrop
   CLIPImageProcessor
   ComputeTimeIds
   MultiAspectRatioResizeCenterCrop
   RandomCrop
   RandomHorizontalFlip
   RandomTextDrop
   SaveImageShape

diffengine.datasets.samplers
===================================

.. contents:: diffengine.datasets.samplers
   :depth: 2
   :local:
   :backlinks: top

.. currentmodule:: diffengine.datasets.samplers

Samplers
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   AspectRatioBatchSampler

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

   ControlNetSaveHook
   IPAdapterSaveHook
   LoRASaveHook
   SDCheckpointHook
   T2IAdapterSaveHook
   UnetEMAHook
   VisualizationHook
