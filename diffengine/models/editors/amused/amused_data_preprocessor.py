import torch
from mmengine.model.base_model.data_preprocessor import BaseDataPreprocessor

from diffengine.registry import MODELS


@MODELS.register_module()
class AMUSEdPreprocessor(BaseDataPreprocessor):
    """AMUSEdPreprocessor."""

    def forward(
            self,
            data: dict,
            training: bool = False  # noqa
    ) -> dict | list:
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
        if "result_class_image" in data["inputs"]:
            # dreambooth with class image
            data["inputs"]["text"] = data["inputs"]["text"] + data["inputs"][
                "result_class_image"].pop("text")
            data["inputs"]["img"] = data["inputs"]["img"] + data["inputs"][
                "result_class_image"].pop("img")
            data["inputs"]["micro_conds"] = data["inputs"]["micro_conds"] + data[
                "inputs"]["result_class_image"].pop("micro_conds")

        data["inputs"]["img"] = torch.stack(data["inputs"]["img"])
        data["inputs"]["micro_conds"] = torch.stack(data["inputs"]["micro_conds"])
        return super().forward(data)
