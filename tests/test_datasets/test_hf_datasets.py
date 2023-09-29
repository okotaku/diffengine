import numpy as np
from mmengine.testing import RunnerTestCase
from PIL import Image

from diffengine.datasets import HFDataset, HFDatasetPreComputeEmbs


class TestHFDataset(RunnerTestCase):

    def test_dataset_from_local(self):
        dataset = HFDataset(
            dataset='tests/testdata/dataset', image_column='file_name')
        assert len(dataset) == 1

        data = dataset[0]
        assert data['text'] == 'a dog'
        self.assertIsInstance(data['img'], Image.Image)
        assert data['img'].width == 400

        dataset = HFDataset(
            dataset='tests/testdata/dataset',
            image_column='file_name',
            csv='metadata2.csv')
        assert len(dataset) == 1

        data = dataset[0]
        assert data['text'] == 'a cat'
        self.assertIsInstance(data['img'], Image.Image)
        assert data['img'].width == 400


class TestHFDatasetPreComputeEmbs(RunnerTestCase):

    def test_dataset_from_local(self):
        dataset = HFDatasetPreComputeEmbs(
            dataset='tests/testdata/dataset',
            model='hf-internal-testing/tiny-stable-diffusion-xl-pipe',
            image_column='file_name')
        assert len(dataset) == 1

        data = dataset[0]
        assert 'text' not in data
        self.assertEqual(type(data['prompt_embeds']), list)
        self.assertEqual(type(data['pooled_prompt_embeds']), list)
        self.assertEqual(np.array(data['prompt_embeds']).shape, (77, 64))
        self.assertEqual(np.array(data['pooled_prompt_embeds']).shape, (32, ))
        self.assertIsInstance(data['img'], Image.Image)
        assert data['img'].width == 400
