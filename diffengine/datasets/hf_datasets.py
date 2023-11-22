# flake8: noqa: TRY004,S311
# yapf: disable
import functools
import gc
import os
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from datasets.fingerprint import Hasher
from mmengine.dataset.base_dataset import Compose
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from diffengine.datasets.utils import encode_prompt_sdxl
from diffengine.models.editors.stable_diffusion_xl.stable_diffusion_xl import (
    import_model_class_from_model_name_or_path,
)
from diffengine.registry import DATASETS

# yapf: enable

Image.MAX_IMAGE_PIXELS = 1000000000


@DATASETS.register_module()
class HFDataset(Dataset):
    """Dataset for huggingface datasets.

    Args:
    ----
        dataset (str): Dataset name or path to dataset.
        image_column (str): Image column name. Defaults to 'image'.
        caption_column (str): Caption column name. Defaults to 'text'.
        csv (str): Caption csv file name when loading local folder.
            Defaults to 'metadata.csv'.
        pipeline (Sequence): Processing pipeline. Defaults to an empty tuple.
        cache_dir (str, optional): The directory where the downloaded datasets
            will be stored.Defaults to None.
    """

    def __init__(self,
                 dataset: str,
                 image_column: str = "image",
                 caption_column: str = "text",
                 csv: str = "metadata.csv",
                 pipeline: Sequence = (),
                 cache_dir: str | None = None) -> None:
        self.dataset_name = dataset
        if Path(dataset).exists():
            # load local folder
            data_file = os.path.join(dataset, csv)
            self.dataset = load_dataset(
                "csv", data_files=data_file, cache_dir=cache_dir)["train"]
        else:
            # load huggingface online
            self.dataset = load_dataset(dataset, cache_dir=cache_dir)["train"]
        self.pipeline = Compose(pipeline)

        self.image_column = image_column
        self.caption_column = caption_column

    def __len__(self) -> int:
        """Get the length of dataset.

        Returns
        -------
            int: The length of filtered dataset.
        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        """Get item.

        Get the idx-th image and data information of dataset after
        ``self.pipeline`.

        Args:
        ----
            idx (int): The index of self.data_list.

        Returns:
        -------
            dict: The idx-th image and data information of dataset after
            ``self.pipeline``.
        """
        data_info = self.dataset[idx]
        image = data_info[self.image_column]
        if isinstance(image, str):
            image = Image.open(os.path.join(self.dataset_name, image))
        image = image.convert("RGB")
        caption = data_info[self.caption_column]
        if isinstance(caption, str):
            pass
        elif isinstance(caption, list | np.ndarray):
            # take a random caption if there are multiple
            caption = random.choice(caption)
        else:
            msg = (f"Caption column `{self.caption_column}` should "
                   "contain either strings or lists of strings.")
            raise ValueError(msg)
        result = {"img": image, "text": caption}
        return self.pipeline(result)


@DATASETS.register_module()
class HFDatasetPreComputeEmbs(HFDataset):
    """Dataset for huggingface datasets.

    The difference from HFDataset is
        1. pre-compute Text Encoder embeddings to save memory.

    Args:
    ----
        model (str): pretrained model name of stable diffusion xl.
            Defaults to 'stabilityai/stable-diffusion-xl-base-1.0'.
        text_hasher (str): Text embeddings hasher name. Defaults to 'text'.
        device (str): Device used to compute embeddings. Defaults to 'cuda'.
        proportion_empty_prompts (float): The probabilities to replace empty
            text. Defaults to 0.9.
    """

    def __init__(self,
                 *args,
                 model: str = "stabilityai/stable-diffusion-xl-base-1.0",
                 text_hasher: str = "text",
                 device: str = "cuda",
                 proportion_empty_prompts: float = 0.0,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        tokenizer_one = AutoTokenizer.from_pretrained(
            model, subfolder="tokenizer", use_fast=False)
        tokenizer_two = AutoTokenizer.from_pretrained(
            model, subfolder="tokenizer_2", use_fast=False)

        text_encoder_cls_one = import_model_class_from_model_name_or_path(
            model)
        text_encoder_cls_two = import_model_class_from_model_name_or_path(
            model, subfolder="text_encoder_2")
        text_encoder_one = text_encoder_cls_one.from_pretrained(
            model, subfolder="text_encoder").to(device)
        text_encoder_two = text_encoder_cls_two.from_pretrained(
            model, subfolder="text_encoder_2").to(device)

        new_fingerprint = Hasher.hash(text_hasher)
        compute_embeddings_fn = functools.partial(
            encode_prompt_sdxl,
            text_encoders=[text_encoder_one, text_encoder_two],
            tokenizers=[tokenizer_one, tokenizer_two],
            proportion_empty_prompts=proportion_empty_prompts,
            caption_column=self.caption_column,
        )
        self.dataset = self.dataset.map(
            compute_embeddings_fn,
            batched=True,
            batch_size=32,
            new_fingerprint=new_fingerprint)

        del text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two
        gc.collect()
        torch.cuda.empty_cache()

    def __getitem__(self, idx: int) -> dict:
        """Get item.

        Get the idx-th image and data information of dataset after
        ``self.train_transforms`.

        Args:
        ----
            idx (int): The index of self.data_list.

        Returns:
        -------
            dict: The idx-th image and data information of dataset after
            ``self.train_transforms``.
        """
        data_info = self.dataset[idx]
        image = data_info[self.image_column]
        if isinstance(image, str):
            image = Image.open(os.path.join(self.dataset_name, image))
        image = image.convert("RGB")
        result = {
            "img": image,
            "prompt_embeds": data_info["prompt_embeds"],
            "pooled_prompt_embeds": data_info["pooled_prompt_embeds"],
        }
        return self.pipeline(result)
