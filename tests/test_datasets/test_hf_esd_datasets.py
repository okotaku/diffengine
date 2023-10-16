import torch
from mmengine.testing import RunnerTestCase

from diffengine.datasets import HFESDDatasetPreComputeEmbs


class TestHFESDDatasetPreComputeEmbs(RunnerTestCase):

    def test_dataset_from_local(self):
        dataset = HFESDDatasetPreComputeEmbs(
            forget_caption="Van Gogh",
            model="hf-internal-testing/tiny-stable-diffusion-xl-pipe",
            device="cpu")
        assert len(dataset) == 1

        data = dataset[0]
        assert "img" not in data
        self.assertEqual(data["text"], "Van Gogh")
        self.assertEqual(type(data["prompt_embeds"]), torch.Tensor)
        self.assertEqual(type(data["pooled_prompt_embeds"]), torch.Tensor)
        self.assertEqual(type(data["null_prompt_embeds"]), torch.Tensor)
        self.assertEqual(type(data["null_pooled_prompt_embeds"]), torch.Tensor)
        self.assertEqual(data["prompt_embeds"].shape, (77, 64))
        self.assertEqual(data["pooled_prompt_embeds"].shape, (32, ))
        self.assertEqual(data["null_prompt_embeds"].shape, (77, 64))
        self.assertEqual(data["null_pooled_prompt_embeds"].shape, (32, ))
