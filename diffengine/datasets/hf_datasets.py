import os
import random
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from datasets import load_dataset
from mmengine.dataset.base_dataset import Compose
from PIL import Image
from torch.utils.data import Dataset

from diffengine.registry import DATASETS


@DATASETS.register_module()
class HFDataset(Dataset):
    """Dataset for huggingface datasets.

    Args:
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
                 image_column: str = 'image',
                 caption_column: str = 'text',
                 csv: str = 'metadata.csv',
                 pipeline: Sequence = (),
                 cache_dir: Optional[str] = None):
        self.dataset_name = dataset
        if Path(dataset).exists():
            # load local folder
            data_file = os.path.join(dataset, csv)
            self.dataset = load_dataset(
                'csv', data_files=data_file, cache_dir=cache_dir)['train']
        else:
            # load huggingface online
            self.dataset = load_dataset(dataset, cache_dir=cache_dir)['train']
        self.pipeline = Compose(pipeline)

        self.image_column = image_column
        self.caption_column = caption_column

    def __len__(self) -> int:
        """Get the length of dataset.

        Returns:
            int: The length of filtered dataset.
        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        """Get the idx-th image and data information of dataset after
        ``self.train_transforms`.

        Args:
            idx (int): The index of self.data_list.

        Returns:
            dict: The idx-th image and data information of dataset after
            ``self.train_transforms``.
        """
        data_info = self.dataset[idx]
        image = data_info[self.image_column]
        if type(image) == str:
            image = Image.open(os.path.join(self.dataset_name, image))
        image = image.convert('RGB')
        caption = data_info[self.caption_column]
        if isinstance(caption, str):
            pass
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            caption = random.choice(caption)
        else:
            raise ValueError(
                f'Caption column `{self.caption_column}` should contain either'
                ' strings or lists of strings.')
        result = dict(img=image, text=caption)
        result = self.pipeline(result)

        return result
