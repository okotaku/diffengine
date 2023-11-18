import copy
import os.path as osp
import shutil
from pathlib import Path

from diffusers.pipelines.wuerstchen import WuerstchenPrior
from mmengine.model import BaseModel
from mmengine.registry import MODELS
from mmengine.testing import RunnerTestCase
from mmengine.testing.runner_test_case import ToyModel
from torch import nn
from transformers import CLIPTextConfig, CLIPTextModel

from diffengine.engine.hooks import WuerstchenSaveHook


class DummyWrapper(BaseModel):

    def __init__(self, model) -> None:
        super().__init__()
        if not isinstance(model, nn.Module):
            model = MODELS.build(model)
        self.module = model

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class ToyModel2(ToyModel):

    def __init__(self) -> None:
        super().__init__()
        model_kwargs = {
            "c_in": 2,
            "c": 8,
            "depth": 2,
            "c_cond": 32,
            "c_r": 8,
            "nhead": 2,
        }

        self.prior = WuerstchenPrior(**model_kwargs)
        config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
        )
        self.text_encoder = CLIPTextModel(config)
        self.finetune_text_encoder = True

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


class TestSDCheckpointHook(RunnerTestCase):

    def setUp(self) -> None:
        MODELS.register_module(name="DummyWrapper", module=DummyWrapper)
        MODELS.register_module(name="ToyModel2", module=ToyModel2)
        return super().setUp()

    def tearDown(self):
        MODELS.module_dict.pop("DummyWrapper")
        MODELS.module_dict.pop("ToyModel2")
        return super().tearDown()

    def test_init(self):
        WuerstchenSaveHook()

    def test_before_save_checkpoint(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model.type = "ToyModel2"
        runner = self.build_runner(cfg)
        checkpoint = dict(state_dict=ToyModel2().state_dict())
        hook = WuerstchenSaveHook()
        hook.before_save_checkpoint(runner, checkpoint)

        assert Path(
            osp.join(runner.work_dir, f"step{runner.iter}", "prior",
                     "diffusion_pytorch_model.safetensors")).exists()
        assert Path(
            osp.join(runner.work_dir, f"step{runner.iter}", "text_encoder",
                     "model.safetensors")).exists()
        shutil.rmtree(osp.join(runner.work_dir), ignore_errors=True)

        for key in checkpoint["state_dict"]:
            assert key.startswith(("prior", "text_encoder"))
