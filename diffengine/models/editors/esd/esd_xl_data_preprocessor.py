from typing import Union

import torch
from mmengine.model.base_model.data_preprocessor import BaseDataPreprocessor

from diffengine.registry import MODELS


@MODELS.register_module()
class ESDXLDataPreprocessor(BaseDataPreprocessor):
    """ESDXLDataPreprocessor."""

    def forward(self,
                data: dict,
                training: bool = False) -> Union[dict, list]:  # noqa
        """Preprocesses the data into the model input format.

        After the data pre-processing of :meth:`cast_data`, ``forward``
        will stack the input tensor list to a batch tensor at the first
        dimension.

        Args:
        ----
            data (dict): Data returned by dataloader
            training (bool): Whether to enable training time augmentation.

        Returns:
        -------
            dict or list: Data in the same format as the model input.
        """
        # pre-compute text embeddings
        data["inputs"]["prompt_embeds"] = torch.stack(
            data["inputs"]["prompt_embeds"])
        data["inputs"]["pooled_prompt_embeds"] = torch.stack(
            data["inputs"]["pooled_prompt_embeds"])
        data["inputs"]["null_prompt_embeds"] = torch.stack(
            data["inputs"]["null_prompt_embeds"])
        data["inputs"]["null_pooled_prompt_embeds"] = torch.stack(
            data["inputs"]["null_pooled_prompt_embeds"])
        return super().forward(data)
