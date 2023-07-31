import shutil

from mmengine.testing import RunnerTestCase
from PIL import Image

from diffengine.datasets import HFDreamBoothDataset


class TestHFDreamBoothDataset(RunnerTestCase):

    def test_dataset(self):
        dataset = HFDreamBoothDataset(
            dataset='diffusers/dog-example',
            instance_prompt='a photo of sks dog')
        assert len(dataset) == 5

        data = dataset[0]
        assert data['text'] == 'a photo of sks dog'
        self.assertIsInstance(data['img'], Image.Image)
        assert data['img'].width == 1815

    def test_dataset_with_class_image(self):
        dataset = HFDreamBoothDataset(
            dataset='diffusers/dog-example',
            instance_prompt='a photo of sks dog',
            class_prompt='a photo of dog',
            class_image_config=dict(
                model='diffusers/tiny-stable-diffusion-torch',
                data_dir='temp_dir/class_image',
                num_images=1,
                device='cpu',
            ),
        )
        assert len(dataset) == 5
        assert len(dataset.class_images) == 1

        data = dataset[0]
        assert data['text'] == 'a photo of sks dog'
        self.assertIsInstance(data['img'], Image.Image)
        assert data['img'].width == 1815

        assert data['result_class_image']['text'] == 'a photo of dog'
        self.assertIsInstance(data['result_class_image']['img'], Image.Image)
        assert data['result_class_image']['img'].width == 128
        shutil.rmtree('temp_dir')
