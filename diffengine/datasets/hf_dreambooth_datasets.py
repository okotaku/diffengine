# flake8: noqa: S311,RUF012
import copy
import hashlib
import os
import random
import shutil
from collections.abc import Sequence
from pathlib import Path

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
    ----
        dataset (str): Dataset name.
        instance_prompt (str):
            The prompt with identifier specifying the instance.
        image_column (str): Image column name. Defaults to 'image'.
        dataset_sub_dir (optional, str): Dataset sub directory name.
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
        csv (str, optional): Image path csv file name when loading local
            folder. If None, the dataset will be loaded from image folders.
            Defaults to None.
        cache_dir (str, optional): The directory where the downloaded datasets
            will be stored.Defaults to None.
    """

    default_class_image_config: dict = {
        "model": "runwayml/stable-diffusion-v1-5",
        "data_dir": "work_dirs/class_image",
        "num_images": 200,
        "device": "cuda",
        "recreate_class_images": True,
    }

    def __init__(self,
                 dataset: str,
                 instance_prompt: str,
                 image_column: str = "image",
                 dataset_sub_dir: str | None = None,
                 class_image_config: dict | None = None,
                 class_prompt: str | None = None,
                 pipeline: Sequence = (),
                 csv: str | None = None,
                 cache_dir: str | None = None) -> None:

        self.dataset_name = dataset
        self.csv = csv

        if class_image_config is None:
            class_image_config = {
                "model": "runwayml/stable-diffusion-v1-5",
                "data_dir": "work_dirs/class_image",
                "num_images": 200,
                "device": "cuda",
                "recreate_class_images": True,
            }
        if Path(dataset).exists():
            # load local folder
            if csv is not None:
                data_file = os.path.join(dataset, csv)
                self.dataset = load_dataset(
                    "csv", data_files=data_file, cache_dir=cache_dir)["train"]
            else:
                self.dataset = load_dataset(dataset, cache_dir=cache_dir)["train"]
        else:  # noqa
            # load huggingface online
            if dataset_sub_dir is not None:
                self.dataset = load_dataset(
                    dataset, dataset_sub_dir, cache_dir=cache_dir)["train"]
            else:
                self.dataset = load_dataset(
                    dataset, cache_dir=cache_dir)["train"]

        self.pipeline = Compose(pipeline)

        self.instance_prompt = instance_prompt
        self.image_column = image_column
        self.class_prompt = class_prompt

        # generate class image
        if class_prompt is not None:
            essential_keys = {
                "model",
                "data_dir",
                "num_images",
                "device",
                "recreate_class_images",
            }
            _class_image_config = copy.deepcopy(
                self.default_class_image_config)
            _class_image_config.update(class_image_config)
            class_image_config = _class_image_config
            assert set(class_image_config ) == essential_keys, \
                f"class_image_config needs a dict with keys {essential_keys}"
            self.generate_class_image(class_image_config)

    def generate_class_image(self, class_image_config: dict) -> None:
        """Generate class images for prior preservation loss."""
        class_images_dir = Path(class_image_config["data_dir"])
        if class_images_dir.exists(
        ) and class_image_config["recreate_class_images"]:
            shutil.rmtree(class_images_dir)
        class_images_dir.mkdir(parents=True, exist_ok=True)
        cur_class_images = list(class_images_dir.iterdir())
        num_cur_images = len(cur_class_images)

        num_new_images = class_image_config["num_images"] - num_cur_images

        pipeline = DiffusionPipeline.from_pretrained(
            class_image_config["model"],
            safety_checker=None,
            torch_dtype=(torch.float16 if class_image_config["device"] != "cpu"
                         else torch.float32),
        )
        pipeline.set_progress_bar_config(disable=True)
        pipeline.to(class_image_config["device"])

        for i in tqdm(range(num_new_images)):
            image = pipeline(self.class_prompt).images[0]
            hash_image = hashlib.sha1(image.tobytes()).hexdigest()  # noqa
            image_filename = (
                class_images_dir / f"{i + num_cur_images}-{hash_image}.jpg")
            image.save(image_filename)
            cur_class_images.append(
                str(image_filename))  # type: ignore[arg-type]

        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.class_images = cur_class_images

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
            if self.csv is not None:
                image = os.path.join(self.dataset_name, image)
            image = Image.open(image)
        image = image.convert("RGB")
        result = {"img": image, "text": self.instance_prompt}
        result = self.pipeline(result)

        if self.class_prompt is not None:
            class_image = random.choice(self.class_images)
            class_image = Image.open(class_image)
            result_class_image = {
                "img": class_image,
                "text": self.class_prompt,
            }
            result_class_image = self.pipeline(result_class_image)
            assert "inputs" in result
            assert "inputs" in result_class_image
            result["inputs"]["result_class_image"] = result_class_image[
                "inputs"]

        return result
