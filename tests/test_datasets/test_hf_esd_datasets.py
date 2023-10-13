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
        assert data["text"] == "Van Gogh"
        assert type(data["prompt_embeds"]) == torch.Tensor
        assert type(data["pooled_prompt_embeds"]) == torch.Tensor
        assert type(data["null_prompt_embeds"]) == torch.Tensor
        assert type(data["null_pooled_prompt_embeds"]) == torch.Tensor
        assert data["prompt_embeds"].shape == (77, 64)
        assert data["pooled_prompt_embeds"].shape == (32, )
        assert data["null_prompt_embeds"].shape == (77, 64)
        assert data["null_pooled_prompt_embeds"].shape == (32, )
