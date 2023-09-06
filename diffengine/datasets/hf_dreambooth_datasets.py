import copy
import hashlib
import random
import shutil
from pathlib import Path
from typing import Optional, Sequence

import torch
from datasets import load_dataset
from diffusers import DiffusionPipeline
from mmengine.dataset.base_dataset import Compose
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from diffengine.registry import DATASETS

Image.MAX_IMAGE_PIXELS = 1000000000


@DATASETS.register_module()
class HFDreamBoothDataset(Dataset):
    """DreamBooth Dataset for huggingface datasets.

    Args:
        dataset (str): Dataset name.
        instance_prompt (str):
            The prompt with identifier specifying the instance.
        image_column (str): Image column name. Defaults to 'image'.
        class_image_config (dict):
            model (str): pretrained model name of stable diffusion to
                create training data of class images.
                Defaults to 'runwayml/stable-diffusion-v1-5'.
            data_dir (str): A folder containing the training data of class
                images. Defaults to 'work_dirs/class_image'.
            num_images (int): Minimal class images for prior preservation
                loss. If there are not enough images already present in
                class_data_dir, additional images will be sampled with
                class_prompt. Defaults to 200.
            recreate_class_images (bool): Whether to re create all class
                images. Defaults to True.
        class_prompt (Optional[str]): The prompt to specify images in the same
                class as provided instance images. Defaults to None.
        pipeline (Sequence): Processing pipeline. Defaults to an empty tuple.
        cache_dir (str, optional): The directory where the downloaded datasets
            will be stored.Defaults to None.
    """
    default_class_image_config: dict = dict(
        model='runwayml/stable-diffusion-v1-5',
        data_dir='work_dirs/class_image',
        num_images=200,
        device='cuda',
        recreate_class_images=True,
    )

    def __init__(self,
                 dataset: str,
                 instance_prompt: str,
                 image_column: str = 'image',
                 class_image_config: dict = dict(
                     model='runwayml/stable-diffusion-v1-5',
                     data_dir='work_dirs/class_image',
                     num_images=200,
                     device='cuda',
                     recreate_class_images=True,
                 ),
                 class_prompt: Optional[str] = None,
                 pipeline: Sequence = (),
                 cache_dir: Optional[str] = None):

        if Path(dataset).exists():
            # load local folder
            data_files = {}
            data_files['train'] = '**'
            self.dataset = load_dataset(
                dataset, data_files, cache_dir=cache_dir)['train']
        else:
            # load huggingface online
            self.dataset = load_dataset(dataset, cache_dir=cache_dir)['train']
        self.pipeline = Compose(pipeline)

        self.instance_prompt = instance_prompt
        self.image_column = image_column
        self.class_prompt = class_prompt

        # generate class image
        if class_prompt is not None:
            essential_keys = {
                'model', 'data_dir', 'num_images', 'device',
                'recreate_class_images'
            }
            _class_image_config = copy.deepcopy(
                self.default_class_image_config)
            _class_image_config.update(class_image_config)
            class_image_config = _class_image_config
            assert isinstance(
                class_image_config, dict
                              ) and set(
                                  class_image_config
                                  ) == essential_keys, \
                f'class_image_config needs a dict with keys {essential_keys}'
            self.generate_class_image(class_image_config)

    def generate_class_image(self, class_image_config):
        class_images_dir = Path(class_image_config['data_dir'])
        if class_images_dir.exists(
        ) and class_image_config['recreate_class_images']:
            shutil.rmtree(class_images_dir)
        class_images_dir.mkdir(parents=True, exist_ok=True)
        cur_class_images = list(class_images_dir.iterdir())
        num_cur_images = len(cur_class_images)

        num_new_images = class_image_config['num_images'] - num_cur_images

        pipeline = DiffusionPipeline.from_pretrained(
            class_image_config['model'],
            safety_checker=None,
        )
        pipeline.set_progress_bar_config(disable=True)
        pipeline.to(class_image_config['device'])

        for i in tqdm(range(num_new_images)):
            image = pipeline(self.class_prompt).images[0]
            hash_image = hashlib.sha1(image.tobytes()).hexdigest()
            image_filename = (
                class_images_dir / f'{i + num_cur_images}-{hash_image}.jpg')
            image.save(image_filename)
            cur_class_images.append(str(image_filename))

        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.class_images = cur_class_images

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
            image = Image.open(image)
        image = image.convert('RGB')
        result = dict(img=image, text=self.instance_prompt)
        result = self.pipeline(result)

        if self.class_prompt is not None:
            class_image = random.choice(self.class_images)
            class_image = Image.open(class_image)
            result_class_image = dict(img=class_image, text=self.class_prompt)
            result_class_image = self.pipeline(result_class_image)
            assert 'inputs' in result
            assert 'inputs' in result_class_image
            result['inputs']['result_class_image'] = result_class_image[
                'inputs']

        return result
