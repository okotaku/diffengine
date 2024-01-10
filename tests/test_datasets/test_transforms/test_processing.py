import os.path as osp
from unittest import TestCase

import numpy as np
import pytest
import torch
import torchvision
from mmengine.dataset.base_dataset import Compose
from mmengine.utils import digit_version
from PIL import Image
from torchvision import transforms

from diffengine.datasets.transforms import TorchVisonTransformWrapper
from diffengine.datasets.transforms.processing import VISION_TRANSFORMS
from diffengine.registry import TRANSFORMS


class TestVisionTransformWrapper(TestCase):

    def test_register(self):
        for t in VISION_TRANSFORMS:
            assert "torchvision/" in t
            assert t in TRANSFORMS

    def test_transform(self):
        img_path = osp.join(osp.dirname(__file__), "../../testdata/color.jpg")
        data = {"img": Image.open(img_path)}

        # test normal transform
        vision_trans = transforms.RandomResizedCrop(224)
        vision_transformed_img = vision_trans(data["img"])
        trans = TRANSFORMS.build(
            dict(type=TorchVisonTransformWrapper,
                 transform=torchvision.transforms.RandomResizedCrop,
                 size=224))
        transformed_img = trans(data)["img"]
        np.equal(np.array(vision_transformed_img), np.array(transformed_img))

        # test convert type dtype
        data = {"img": torch.randn(3, 224, 224)}
        vision_trans = transforms.ConvertImageDtype(torch.float)
        vision_transformed_img = vision_trans(data["img"])
        trans = TRANSFORMS.build(
            dict(type=TorchVisonTransformWrapper,
                 transform=torchvision.transforms.ConvertImageDtype,
                 dtype="float"))
        transformed_img = trans(data)["img"]
        np.equal(np.array(vision_transformed_img), np.array(transformed_img))

        # test transform with interpolation
        data = {"img": Image.open(img_path)}
        if digit_version(torchvision.__version__) > digit_version("0.8.0"):
            from torchvision.transforms import InterpolationMode
            interpolation_t = InterpolationMode.NEAREST
        else:
            interpolation_t = Image.NEAREST
        vision_trans = transforms.Resize(224, interpolation_t)
        vision_transformed_img = vision_trans(data["img"])
        trans = TRANSFORMS.build(
            dict(type=TorchVisonTransformWrapper,
                 transform=torchvision.transforms.Resize,
                 size=224, interpolation="nearest"))
        transformed_img = trans(data)["img"]
        np.equal(np.array(vision_transformed_img), np.array(transformed_img))

        # test compose transforms
        data = {"img": Image.open(img_path)}
        vision_trans = transforms.Compose([
            transforms.Resize(176),
            transforms.RandomHorizontalFlip(),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        vision_transformed_img = vision_trans(data["img"])

        pipeline_cfg = [
            dict(type=TorchVisonTransformWrapper,
                 transform=torchvision.transforms.Resize,
                 size=176),
            dict(type="RandomHorizontalFlip"),
            dict(type=TorchVisonTransformWrapper,
                 transform=torchvision.transforms.PILToTensor),
            dict(type=TorchVisonTransformWrapper,
                 transform=torchvision.transforms.ConvertImageDtype,
                 dtype="float"),
            dict(
                type=TorchVisonTransformWrapper,
                transform=torchvision.transforms.Normalize,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
        pipeline = [TRANSFORMS.build(t) for t in pipeline_cfg]
        pipe = Compose(transforms=pipeline)
        transformed_img = pipe(data)["img"]
        np.equal(np.array(vision_transformed_img), np.array(transformed_img))


class TestSaveImageShape(TestCase):

    def test_register(self):
        assert "SaveImageShape" in TRANSFORMS

    def test_transform(self):
        img_path = osp.join(osp.dirname(__file__), "../../testdata/color.jpg")
        data = {"img": Image.open(img_path)}
        ori_img_shape = [data["img"].height, data["img"].width]

        # test transform
        trans = TRANSFORMS.build(dict(type="SaveImageShape"))
        data = trans(data)
        self.assertListEqual(data["ori_img_shape"], ori_img_shape)


class TestComputeTimeIds(TestCase):

    def test_register(self):
        assert "ComputeTimeIds" in TRANSFORMS

    def test_transform(self):
        img_path = osp.join(osp.dirname(__file__), "../../testdata/color.jpg")
        img = Image.open(img_path)
        data = {"img": img, "ori_img_shape": [32, 32], "crop_top_left": [0, 0]}

        # test transform
        trans = TRANSFORMS.build(dict(type="ComputeTimeIds"))
        data = trans(data)
        self.assertListEqual(data["time_ids"],
                             [32, 32, 0, 0, img.height, img.width])


class TestRandomCrop(TestCase):
    crop_size = 32

    def test_register(self):
        assert "RandomCrop" in TRANSFORMS

    def test_transform(self):
        img_path = osp.join(osp.dirname(__file__), "../../testdata/color.jpg")
        data = {"img": Image.open(img_path)}

        # test transform
        trans = TRANSFORMS.build(dict(type="RandomCrop", size=self.crop_size))
        data = trans(data)
        assert "crop_top_left" in data
        assert len(data["crop_top_left"]) == 2
        assert data["img"].height == data["img"].width == self.crop_size
        upper, left = data["crop_top_left"]
        lower, right = data["crop_bottom_right"]
        assert lower == upper + self.crop_size
        assert right == left + self.crop_size
        np.equal(
            np.array(data["img"]),
            np.array(Image.open(img_path).crop((left, upper, right, lower))))

    def test_transform_multiple_keys(self):
        img_path = osp.join(osp.dirname(__file__), "../../testdata/color.jpg")
        data = {
            "img": Image.open(img_path),
            "condition_img": Image.open(img_path),
        }

        # test transform
        trans = TRANSFORMS.build(
            dict(
                type="RandomCrop",
                size=self.crop_size,
                keys=["img", "condition_img"]))
        data = trans(data)
        assert "crop_top_left" in data
        assert len(data["crop_top_left"]) == 2
        assert data["img"].height == data["img"].width == self.crop_size
        upper, left = data["crop_top_left"]
        lower, right = data["crop_bottom_right"]
        assert lower == upper + self.crop_size
        assert right == left + self.crop_size
        np.equal(
            np.array(data["img"]),
            np.array(Image.open(img_path).crop((left, upper, right, lower))))
        np.equal(np.array(data["img"]), np.array(data["condition_img"]))

        # size mismatch
        data = {
            "img": Image.open(img_path),
            "condition_img": Image.open(img_path).resize((298, 398)),
        }
        with pytest.raises(
                AssertionError, match="Size mismatch"):
            data = trans(data)

        # test transform force_same_size=False
        trans = TRANSFORMS.build(
            dict(
                type="RandomCrop",
                size=self.crop_size,
                force_same_size=False,
                keys=["img", "condition_img"]))
        data = trans(data)
        assert "crop_top_left" in data
        assert len(data["crop_top_left"]) == 2
        assert data["img"].height == data["img"].width == self.crop_size
        upper, left = data["crop_top_left"]
        lower, right = data["crop_bottom_right"]
        assert lower == upper + self.crop_size
        assert right == left + self.crop_size


class TestCenterCrop(TestCase):
    crop_size = 32

    def test_register(self):
        assert "CenterCrop" in TRANSFORMS

    def test_transform(self):
        img_path = osp.join(osp.dirname(__file__), "../../testdata/color.jpg")
        data = {"img": Image.open(img_path)}

        # test transform
        trans = TRANSFORMS.build(dict(type="CenterCrop", size=self.crop_size))
        data = trans(data)
        assert "crop_top_left" in data
        assert len(data["crop_top_left"]) == 2
        assert data["img"].height == data["img"].width == self.crop_size
        upper, left = data["crop_top_left"]
        lower, right = data["crop_bottom_right"]
        assert lower == upper + self.crop_size
        assert right == left + self.crop_size
        np.equal(
            np.array(data["img"]),
            np.array(Image.open(img_path).crop((left, upper, right, lower))))

    def test_transform_multiple_keys(self):
        img_path = osp.join(osp.dirname(__file__), "../../testdata/color.jpg")
        data = {
            "img": Image.open(img_path),
            "condition_img": Image.open(img_path),
        }

        # test transform
        trans = TRANSFORMS.build(
            dict(
                type="CenterCrop",
                size=self.crop_size,
                keys=["img", "condition_img"]))
        data = trans(data)
        assert "crop_top_left" in data
        assert len(data["crop_top_left"]) == 2
        assert data["img"].height == data["img"].width == self.crop_size
        upper, left = data["crop_top_left"]
        lower, right = data["crop_bottom_right"]
        assert lower == upper + self.crop_size
        assert right == left + self.crop_size
        np.equal(
            np.array(data["img"]),
            np.array(Image.open(img_path).crop((left, upper, right, lower))))
        np.equal(np.array(data["img"]), np.array(data["condition_img"]))


class TestRandomHorizontalFlip(TestCase):

    def test_register(self):
        assert "RandomHorizontalFlip" in TRANSFORMS

    def test_transform(self):
        img_path = osp.join(osp.dirname(__file__), "../../testdata/color.jpg")
        data = {
            "img": Image.open(img_path),
            "crop_top_left": [0, 0],
            "crop_bottom_right": [10, 10],
        }

        # test transform
        trans = TRANSFORMS.build(dict(type="RandomHorizontalFlip", p=1.))
        data = trans(data)
        assert "crop_top_left" in data
        assert len(data["crop_top_left"]) == 2
        self.assertListEqual(data["crop_top_left"],
                             [0, data["img"].width - 10])

        np.equal(
            np.array(data["img"]),
            np.array(Image.open(img_path).transpose(Image.FLIP_LEFT_RIGHT)))

        # test transform p=0.0
        data = {
            "img": Image.open(img_path),
            "crop_top_left": [0, 0],
            "crop_bottom_right": [10, 10],
        }
        trans = TRANSFORMS.build(dict(type="RandomHorizontalFlip", p=0.))
        data = trans(data)
        assert "crop_top_left" in data
        self.assertListEqual(data["crop_top_left"], [0, 0])

        np.equal(np.array(data["img"]), np.array(Image.open(img_path)))

    def test_transform_multiple_keys(self):
        img_path = osp.join(osp.dirname(__file__), "../../testdata/color.jpg")
        data = {
            "img": Image.open(img_path),
            "condition_img": Image.open(img_path),
            "crop_top_left": [0, 0],
            "crop_bottom_right": [10, 10],
        }

        # test transform
        trans = TRANSFORMS.build(
            dict(
                type="RandomHorizontalFlip",
                p=1.,
                keys=["img", "condition_img"]))
        data = trans(data)
        assert "crop_top_left" in data
        assert len(data["crop_top_left"]) == 2
        self.assertListEqual(data["crop_top_left"],
                             [0, data["img"].width - 10])

        np.equal(
            np.array(data["img"]),
            np.array(Image.open(img_path).transpose(Image.FLIP_LEFT_RIGHT)))
        np.equal(np.array(data["img"]), np.array(data["condition_img"]))


class TestMultiAspectRatioResizeCenterCrop(TestCase):
    sizes = [(32, 32), (16, 48)]  # noqa

    def test_register(self):
        assert "MultiAspectRatioResizeCenterCrop" in TRANSFORMS

    def test_transform(self):
        img_path = osp.join(osp.dirname(__file__), "../../testdata/color.jpg")
        data = {"img": Image.open(img_path).resize((32, 36))}

        # test transform
        trans = TRANSFORMS.build(
            dict(type="MultiAspectRatioResizeCenterCrop", sizes=self.sizes))
        data = trans(data)
        assert "crop_top_left" in data
        assert len(data["crop_top_left"]) == 2
        self.assertTupleEqual((data["img"].height, data["img"].width),
                              self.sizes[0])
        upper, left = data["crop_top_left"]
        lower, right = data["crop_bottom_right"]
        assert lower == upper + self.sizes[0][0]
        assert right == left + self.sizes[0][1]
        np.equal(
            np.array(data["img"]),
            np.array(
                Image.open(img_path).resize((32, 36)).crop(
                    (left, upper, right, lower))))

        # test 2nd size
        data = {"img": Image.open(img_path).resize((55, 16))}
        data = trans(data)
        assert "crop_top_left" in data
        assert len(data["crop_top_left"]) == 2
        self.assertTupleEqual((data["img"].height, data["img"].width),
                              self.sizes[1])
        upper, left = data["crop_top_left"]
        lower, right = data["crop_bottom_right"]
        assert lower == upper + self.sizes[1][0]
        assert right == left + self.sizes[1][1]
        np.equal(
            np.array(data["img"]),
            np.array(
                Image.open(img_path).resize((55, 16)).crop(
                    (left, upper, right, lower))))

    def test_transform_multiple_keys(self):
        img_path = osp.join(osp.dirname(__file__), "../../testdata/color.jpg")
        data = {
            "img": Image.open(img_path).resize((32, 36)),
            "condition_img": Image.open(img_path).resize((32, 36)),
        }

        # test transform
        trans = TRANSFORMS.build(
            dict(
                type="MultiAspectRatioResizeCenterCrop",
                sizes=self.sizes,
                keys=["img", "condition_img"]))
        data = trans(data)
        assert "crop_top_left" in data
        assert len(data["crop_top_left"]) == 2
        self.assertTupleEqual((data["img"].height, data["img"].width),
                              self.sizes[0])
        upper, left = data["crop_top_left"]
        lower, right = data["crop_bottom_right"]
        assert lower == upper + self.sizes[0][0]
        assert right == left + self.sizes[0][1]
        np.equal(
            np.array(data["img"]),
            np.array(
                Image.open(img_path).resize((32, 36)).crop(
                    (left, upper, right, lower))))
        np.equal(np.array(data["img"]), np.array(data["condition_img"]))


class TestCLIPImageProcessor(TestCase):

    def test_register(self):
        assert "CLIPImageProcessor" in TRANSFORMS

    def test_transform(self):
        img_path = osp.join(osp.dirname(__file__), "../../testdata/color.jpg")
        data = {
            "img": Image.open(img_path),
        }

        # test transform
        trans = TRANSFORMS.build(dict(type="CLIPImageProcessor"))
        data = trans(data)
        assert "clip_img" in data
        assert type(data["clip_img"]) == torch.Tensor
        assert data["clip_img"].size() == (3, 224, 224)


class TestRandomTextDrop(TestCase):

    def test_register(self):
        assert "RandomTextDrop" in TRANSFORMS

    def test_transform(self):
        data = {
            "text": "a dog",
        }

        # test transform
        trans = TRANSFORMS.build(dict(type="RandomTextDrop", p=1.))
        data = trans(data)
        assert data["text"] == ""

        # test transform p=0.0
        data = {
            "text": "a dog",
        }
        trans = TRANSFORMS.build(dict(type="RandomTextDrop", p=0.))
        data = trans(data)
        assert data["text"] == "a dog"


class TestComputePixArtImgInfo(TestCase):

    def test_register(self):
        assert "ComputePixArtImgInfo" in TRANSFORMS

    def test_transform(self):
        img_path = osp.join(osp.dirname(__file__), "../../testdata/color.jpg")
        img = Image.open(img_path)
        data = {"img": img, "ori_img_shape": [32, 32], "crop_top_left": [0, 0]}

        # test transform
        trans = TRANSFORMS.build(dict(type="ComputePixArtImgInfo"))
        data = trans(data)
        self.assertListEqual(data["resolution"],
                             [float(d) for d in data["ori_img_shape"]])
        assert data["aspect_ratio"] == img.height / img.width


class TestT5TextPreprocess(TestCase):

    def test_register(self):
        assert "T5TextPreprocess" in TRANSFORMS

    def test_transform(self):
        data = {
            "text": "A dog",
        }

        # test transform
        trans = TRANSFORMS.build(dict(type="T5TextPreprocess"))
        data = trans(data)
        assert data["text"] == "a dog"

        data = {
            "text": "A dog in https://dummy.dummy",
        }
        data = trans(data)
        assert data["text"] == "a dog in dummy. dummy"


class TestMaskToTensor(TestCase):

    def test_register(self):
        assert "MaskToTensor" in TRANSFORMS

    def test_transform(self):
        data = {"mask": np.zeros((32, 32, 1))}

        # test transform
        trans = TRANSFORMS.build(dict(type="MaskToTensor"))
        data = trans(data)
        assert data["mask"].shape == (1, 32, 32)


class TestGetMaskedImage(TestCase):

    def test_register(self):
        assert "GetMaskedImage" in TRANSFORMS

    def test_transform(self):
        img_path = osp.join(osp.dirname(__file__), "../../testdata/color.jpg")
        img = torch.Tensor(np.array(Image.open(img_path)))
        mask = np.zeros((img.shape[0], img.shape[1], 1))
        mask[:10, :10] = 1
        mask = torch.Tensor(mask)
        data = {"img": img, "mask": mask}

        # test transform
        trans = TRANSFORMS.build(dict(type="GetMaskedImage"))
        data = trans(data)
        assert "masked_image" in data
        assert data["masked_image"].shape == img.shape
        assert torch.allclose(data["masked_image"][10:, 10:], img[10:, 10:])
        assert data["masked_image"][:10, :10].sum() == 0


class TestAddConstantCaption(TestCase):

    def test_register(self):
        assert "AddConstantCaption" in TRANSFORMS

    def test_transform(self):
        data = {
            "text": "a dog.",
        }

        # test transform
        trans = TRANSFORMS.build(dict(type="AddConstantCaption",
                                      constant_caption="in szn style"))
        data = trans(data)
        assert data["text"] == "a dog. in szn style"
