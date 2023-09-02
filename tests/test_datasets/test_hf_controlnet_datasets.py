from mmengine.testing import RunnerTestCase
from PIL import Image

from diffengine.datasets import HFControlNetDataset


class TestHFControlNetDataset(RunnerTestCase):

    def test_dataset_from_local(self):
        dataset = HFControlNetDataset(
            dataset='tests/testdata/dataset',
            image_column='file_name',
            csv='metadata_cn.csv')
        assert len(dataset) == 1

        data = dataset[0]
        assert data['text'] == 'a dog'
        self.assertIsInstance(data['img'], Image.Image)
        assert data['img'].width == 400
        self.assertIsInstance(data['img'], Image.Image)
        assert data['img'].width == 400
        self.assertIsInstance(data['condition_img'], Image.Image)
        assert data['condition_img'].width == 400
