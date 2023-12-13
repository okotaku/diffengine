import shutil

from mmengine.registry import TRANSFORMS
from mmengine.testing import RunnerTestCase
from PIL import Image

from diffengine.datasets import HFDreamBoothDataset
from diffengine.datasets.transforms import PackInputs


class TestHFDreamBoothDataset(RunnerTestCase):

    def setUp(self) -> None:
        TRANSFORMS.register_module(
            name="PackInputs", module=PackInputs, force=True)
        return super().setUp()

    def tearDown(self):
        TRANSFORMS.module_dict.pop("PackInputs")
        return super().tearDown()

    def test_dataset(self):
        dataset = HFDreamBoothDataset(
            dataset="diffusers/dog-example",
            instance_prompt="a photo of sks dog")
        assert len(dataset) == 5

        data = dataset[0]
        assert data["text"] == "a photo of sks dog"
        assert isinstance(data["img"], Image.Image)
        assert data["img"].width == 1815

    def test_dataset_with_class_image(self):
        dataset = HFDreamBoothDataset(
            dataset="diffusers/dog-example",
            instance_prompt="a photo of sks dog",
            class_prompt="a photo of dog",
            class_image_config=dict(
                model="diffusers/tiny-stable-diffusion-torch",
                data_dir="temp_dir/class_image",
                num_images=1,
                device="cpu",
                recreate_class_images=True,
            ),
            pipeline=[
                dict(type="PackInputs", skip_to_tensor_key=["img", "text"]),
            ])
        assert len(dataset) == 5
        assert len(dataset.class_images) == 1

        data = dataset[0]
        assert data["inputs"]["text"] == "a photo of sks dog"
        assert isinstance(data["inputs"]["img"], Image.Image)
        assert data["inputs"]["img"].width == 1815

        assert data["inputs"]["result_class_image"]["text"] == "a photo of dog"
        assert isinstance(data["inputs"]["result_class_image"]["img"],
                          Image.Image)
        assert data["inputs"]["result_class_image"]["img"].width == 128
        shutil.rmtree("temp_dir")

    def test_dataset_from_local(self):
        dataset = HFDreamBoothDataset(
            dataset="tests/testdata/dataset_db",
            instance_prompt="a photo of sks dog")
        assert len(dataset) == 1

        data = dataset[0]
        assert data["text"] == "a photo of sks dog"
        assert isinstance(data["img"], Image.Image)
        assert data["img"].width == 400

    def test_dataset_from_local_with_csv(self):
        dataset = HFDreamBoothDataset(
            dataset="tests/testdata/dataset",
            csv="metadata.csv",
            image_column="file_name",
            instance_prompt="a photo of sks dog")
        assert len(dataset) == 1

        data = dataset[0]
        assert data["text"] == "a photo of sks dog"
        assert isinstance(data["img"], Image.Image)
        assert data["img"].width == 400
