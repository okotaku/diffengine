import random

import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision import transforms

from diffengine.registry import DATASETS


@DATASETS.register_module()
class HFDataset(Dataset):
    """Dataset for huggingface datasets.

    Args:
        dataset (str): Dataset name.
        image_column (str): Image column name. Defaults to 'image'.
        caption_column (str): Caption column name. Defaults to 'text'.
        resolution (int): Input image size. Defaults to 512.
        center_crop (bool): If true, use center crop.
            If false, use random crop. Defaults to True.
        random_flip (bool): If true, use random flip. Defaults to True.
    """

    def __init__(self,
                 dataset: str,
                 image_column: str = 'image',
                 caption_column: str = 'text',
                 resolution: int = 512,
                 center_crop: bool = True,
                 random_flip: bool = True):

        self.dataset = load_dataset(dataset)['train']
        self.train_transforms = transforms.Compose([
            transforms.Resize(
                resolution,
                interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution)
            if center_crop else transforms.RandomCrop(resolution),
            transforms.RandomHorizontalFlip()
            if random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

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
        image = data_info[self.image_column].convert('RGB')
        image = self.train_transforms(image)
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

        return dict(pixel_values=image, text=caption)
