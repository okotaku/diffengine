import os.path as osp
from unittest import TestCase

import numpy as np
import torch
import torchvision
from mmengine.dataset.base_dataset import Compose
from mmengine.utils import digit_version
from PIL import Image
from torchvision import transforms

from diffengine.datasets.transforms.processing import VISION_TRANSFORMS
from diffengine.registry import TRANSFORMS


class TestVisionTransformWrapper(TestCase):

    def test_register(self):
        for t in VISION_TRANSFORMS:
            self.assertIn('torchvision/', t)
            self.assertIn(t, TRANSFORMS)

    def test_transform(self):
        img_path = osp.join(osp.dirname(__file__), '../../data/color.jpg')
        data = {'img': Image.open(img_path)}

        # test normal transform
        vision_trans = transforms.RandomResizedCrop(224)
        vision_transformed_img = vision_trans(data['img'])
        trans = TRANSFORMS.build(
            dict(type='torchvision/RandomResizedCrop', size=224))
        transformed_img = trans(data)['img']
        np.equal(np.array(vision_transformed_img), np.array(transformed_img))

        # test convert type dtype
        data = {'img': torch.randn(3, 224, 224)}
        vision_trans = transforms.ConvertImageDtype(torch.float)
        vision_transformed_img = vision_trans(data['img'])
        trans = TRANSFORMS.build(
            dict(type='torchvision/ConvertImageDtype', dtype='float'))
        transformed_img = trans(data)['img']
        np.equal(np.array(vision_transformed_img), np.array(transformed_img))

        # test transform with interpolation
        data = {'img': Image.open(img_path)}
        if digit_version(torchvision.__version__) > digit_version('0.8.0'):
            from torchvision.transforms import InterpolationMode
            interpolation_t = InterpolationMode.NEAREST
        else:
            interpolation_t = Image.NEAREST
        vision_trans = transforms.Resize(224, interpolation_t)
        vision_transformed_img = vision_trans(data['img'])
        trans = TRANSFORMS.build(
            dict(type='torchvision/Resize', size=224, interpolation='nearest'))
        transformed_img = trans(data)['img']
        np.equal(np.array(vision_transformed_img), np.array(transformed_img))

        # test compose transforms
        data = {'img': Image.open(img_path)}
        vision_trans = transforms.Compose([
            transforms.Resize(176),
            transforms.RandomHorizontalFlip(),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        vision_transformed_img = vision_trans(data['img'])

        pipeline_cfg = [
            dict(type='torchvision/Resize', size=176),
            dict(type='torchvision/RandomHorizontalFlip'),
            dict(type='torchvision/PILToTensor'),
            dict(type='torchvision/ConvertImageDtype', dtype='float'),
            dict(
                type='torchvision/Normalize',
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
        ]
        pipeline = [TRANSFORMS.build(t) for t in pipeline_cfg]
        pipe = Compose(transforms=pipeline)
        transformed_img = pipe(data)['img']
        np.equal(np.array(vision_transformed_img), np.array(transformed_img))


class TestSaveImageShape(TestCase):

    def test_register(self):
        self.assertIn('SaveImageShape', TRANSFORMS)

    def test_transform(self):
        img_path = osp.join(osp.dirname(__file__), '../../data/color.jpg')
        data = {'img': Image.open(img_path)}
        ori_img_shape = [data['img'].height, data['img'].width]

        # test transform
        trans = TRANSFORMS.build(dict(type='SaveImageShape'))
        data = trans(data)
        self.assertListEqual(data['ori_img_shape'], ori_img_shape)


class TestComputeTimeIds(TestCase):

    def test_register(self):
        self.assertIn('ComputeTimeIds', TRANSFORMS)

    def test_transform(self):
        img_path = osp.join(osp.dirname(__file__), '../../data/color.jpg')
        img = Image.open(img_path)
        data = {'img': img, 'ori_img_shape': [32, 32], 'crop_top_left': [0, 0]}

        # test transform
        trans = TRANSFORMS.build(dict(type='ComputeTimeIds'))
        data = trans(data)
        self.assertListEqual(data['time_ids'],
                             [32, 32, 0, 0, img.height, img.width])


class TestRandomCropWithCropPoint(TestCase):
    crop_size = 32

    def test_register(self):
        self.assertIn('RandomCropWithCropPoint', TRANSFORMS)

    def test_transform(self):
        img_path = osp.join(osp.dirname(__file__), '../../data/color.jpg')
        data = {'img': Image.open(img_path)}

        # test transform
        trans = TRANSFORMS.build(
            dict(type='RandomCropWithCropPoint', size=self.crop_size))
        data = trans(data)
        self.assertIn('crop_top_left', data)
        assert len(data['crop_top_left']) == 2
        assert data['img'].height == data['img'].width == self.crop_size
        upper, left = data['crop_top_left']
        right = left + self.crop_size
        lower = upper + self.crop_size
        np.equal(
            np.array(data['img']),
            np.array(Image.open(img_path).crop((left, upper, right, lower))))


class TestCenterCropWithCropPoint(TestCase):
    crop_size = 32

    def test_register(self):
        self.assertIn('CenterCropWithCropPoint', TRANSFORMS)

    def test_transform(self):
        img_path = osp.join(osp.dirname(__file__), '../../data/color.jpg')
        data = {'img': Image.open(img_path)}

        # test transform
        trans = TRANSFORMS.build(
            dict(type='CenterCropWithCropPoint', size=self.crop_size))
        data = trans(data)
        self.assertIn('crop_top_left', data)
        assert len(data['crop_top_left']) == 2
        assert data['img'].height == data['img'].width == self.crop_size
        upper, left = data['crop_top_left']
        right = left + self.crop_size
        lower = upper + self.crop_size
        np.equal(
            np.array(data['img']),
            np.array(Image.open(img_path).crop((left, upper, right, lower))))


class TestRandomHorizontalFlipFixCropPoint(TestCase):

    def test_register(self):
        self.assertIn('RandomHorizontalFlipFixCropPoint', TRANSFORMS)

    def test_transform(self):
        img_path = osp.join(osp.dirname(__file__), '../../data/color.jpg')
        data = {'img': Image.open(img_path), 'crop_top_left': [0, 0]}

        # test transform
        trans = TRANSFORMS.build(
            dict(type='RandomHorizontalFlipFixCropPoint', p=1.))
        data = trans(data)
        self.assertIn('crop_top_left', data)
        assert len(data['crop_top_left']) == 2
        self.assertListEqual(data['crop_top_left'], [0, data['img'].width])

        np.equal(
            np.array(data['img']),
            np.array(Image.open(img_path).transpose(Image.FLIP_LEFT_RIGHT)))

        # test transform p=0.0
        data = {'img': Image.open(img_path), 'crop_top_left': [0, 0]}
        trans = TRANSFORMS.build(
            dict(type='RandomHorizontalFlipFixCropPoint', p=0.))
        data = trans(data)
        self.assertIn('crop_top_left', data)
        self.assertListEqual(data['crop_top_left'], [0, 0])

        np.equal(np.array(data['img']), np.array(Image.open(img_path)))
