import gc
from collections.abc import Sequence

import torch
from mmengine.dataset.base_dataset import Compose
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)

from diffengine.datasets.utils import encode_prompt_sdxl
from diffengine.registry import DATASETS

Image.MAX_IMAGE_PIXELS = 1000000000


@DATASETS.register_module()
class HFESDDatasetPreComputeEmbs(Dataset):
    """Huggingface Erasing Concepts from Diffusion Models Dataset.

    Dataset of huggingface datasets for Erasing Concepts from Diffusion
    Models.

    Args:
    ----
        forget_caption (str): The caption used to forget.
        model (str): pretrained model name of stable diffusion xl.
            Defaults to 'stabilityai/stable-diffusion-xl-base-1.0'.
        device (str): Device used to compute embeddings. Defaults to 'cuda'.
        pipeline (Sequence): Processing pipeline. Defaults to an empty tuple.
    """

    def __init__(self,
                 forget_caption: str,
                 model: str = "stabilityai/stable-diffusion-xl-base-1.0",
                 device: str = "cuda",
                 pipeline: Sequence = ()) -> None:
        self.pipeline = Compose(pipeline)
        self.forget_caption = forget_caption

        tokenizer_one = AutoTokenizer.from_pretrained(
            model, subfolder="tokenizer", use_fast=False)
        tokenizer_two = AutoTokenizer.from_pretrained(
            model, subfolder="tokenizer_2", use_fast=False)

        text_encoder_one = CLIPTextModel.from_pretrained(
            model, subfolder="text_encoder").to(device)
        text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
            model, subfolder="text_encoder_2").to(device)

        # null prompt
        self.embs = encode_prompt_sdxl(
            {"text": [self.forget_caption, ""]},
            text_encoders=[text_encoder_one, text_encoder_two],
            tokenizers=[tokenizer_one, tokenizer_two],
            caption_column="text")

        del text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two
        gc.collect()
        torch.cuda.empty_cache()

    def __len__(self) -> int:
        """Get the length of dataset.

        Returns
        -------
            int: The length of filtered dataset.
        """
        return 1

    def __getitem__(self, idx: int) -> dict:
        """Get the dataset after ``self.pipeline`.

        Args:
        ----
            idx (int): The index.

        Returns:
        -------
            dict: The idx-th data information of dataset after
            ``self.pipeline``.
        """
        result = {
            "text": self.forget_caption,
            "prompt_embeds": self.embs["prompt_embeds"][0],
            "pooled_prompt_embeds": self.embs["pooled_prompt_embeds"][0],
            "null_prompt_embeds": self.embs["prompt_embeds"][1],
            "null_pooled_prompt_embeds": self.embs["pooled_prompt_embeds"][1],
        }
        return self.pipeline(result)
