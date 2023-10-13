import copy
from os import path as osp

import cv2
import mmengine
import numpy as np
from torch.multiprocessing import Value

from diffengine.registry import TRANSFORMS


@TRANSFORMS.register_module()
class DumpImage:
    """Dump the image processed by the pipeline.

    Args:
        max_imgs (int): Maximum value of output.
        dump_dir (str): Dump output directory.
    """

    def __init__(self, max_imgs: int, dump_dir: str):
        self.max_imgs = max_imgs
        self.dump_dir = dump_dir
        mmengine.mkdir_or_exist(self.dump_dir)
        self.num_dumped_imgs = Value("i", 0)

    def __call__(self, results):
        """Dump the input image to the specified directory.

        No changes will be
        made.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            results (dict): Result dict from loading pipeline. (same as input)
        """

        enable_dump = False
        with self.num_dumped_imgs.get_lock():
            if self.num_dumped_imgs.value < self.max_imgs:
                self.num_dumped_imgs.value += 1
                enable_dump = True
                dump_id = self.num_dumped_imgs.value

        if enable_dump:
            img = copy.deepcopy(results["img"])
            if img.shape[0] in [1, 3]:
                img = img.permute(1, 2, 0) * 255
            out_file = osp.join(self.dump_dir, f"{dump_id}_image.png")
            cv2.imwrite(out_file, img.numpy().astype(np.uint8)[..., ::-1])

            if "condition_img" in results:
                condition_img = results["condition_img"]
                if condition_img.shape[0] in [1, 3]:
                    condition_img = condition_img.permute(1, 2, 0) * 255
                cond_out_file = osp.join(self.dump_dir, f"{dump_id}_cond.png")
                cv2.imwrite(cond_out_file,
                            condition_img.numpy().astype(np.uint8)[..., ::-1])

        return results
