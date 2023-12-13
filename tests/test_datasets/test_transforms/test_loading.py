import os.path as osp
from unittest import TestCase

import numpy as np
from PIL import Image

from diffengine.registry import TRANSFORMS


class TestLoadMask(TestCase):

    def test_register(self):
        assert "LoadMask" in TRANSFORMS

    def test_transform(self):
        img_path = osp.join(osp.dirname(__file__), "../../testdata/color.jpg")
        img = Image.open(img_path)
        data = {"img": img}

        # test bbox mask
        trans = TRANSFORMS.build(dict(
            type="LoadMask",
            mask_mode="bbox",
            mask_config=dict(
                max_bbox_shape=128)))
        data = trans(data)
        assert "mask" in data
        assert data["mask"].shape == (img.height, img.width, 1)
        assert np.all(np.unique(data["mask"]) == [0, 1])

        # test bbox irregular holes
        data = {"img": img}
        trans = TRANSFORMS.build(dict(
            type="LoadMask",
            mask_mode="irregular",
            mask_config=dict(
                num_vertices=(4, 12),
                max_angle=4.,
                length_range=(10, 100),
                brush_width=(10, 40),
                area_ratio_range=(0.15, 0.5))))
        data = trans(data)
        assert "mask" in data
        assert data["mask"].shape == (img.height, img.width, 1)
        assert np.all(np.unique(data["mask"]) == [0, 1])

        # test ff
        trans = TRANSFORMS.build(dict(
            type="LoadMask",
            mask_mode="ff",
            mask_config=dict(
                num_vertices=(4, 12),
                mean_angle=1.2,
                angle_range=0.4,
                brush_width=(12, 40))))
        data = trans(data)
        assert "mask" in data
        assert data["mask"].shape == (img.height, img.width, 1)
        assert np.all(np.unique(data["mask"]) == [0, 1])

        # test whole
        trans = TRANSFORMS.build(dict(
            type="LoadMask",
            mask_mode="whole"))
        data = trans(data)
        assert "mask" in data
        assert data["mask"].shape == (img.height, img.width, 1)
        assert np.all(np.unique(data["mask"]) == 1)
