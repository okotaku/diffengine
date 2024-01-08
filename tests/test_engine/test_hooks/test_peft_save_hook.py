import copy
import os.path as osp
import shutil
from pathlib import Path

from mmengine.config import Config
from mmengine.model import BaseModel
from mmengine.registry import MODELS
from mmengine.testing import RunnerTestCase
from torch import nn

from diffengine.engine.hooks import PeftSaveHook
from diffengine.models.utils import TimeSteps, WhiteNoise, WuerstchenRandomTimeSteps


class DummyWrapper(BaseModel):

    def __init__(self, model) -> None:
        super().__init__()
        if not isinstance(model, nn.Module):
            model = MODELS.build(model)
        self.module = model

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class TestPeftSaveHook(RunnerTestCase):

    def setUp(self) -> None:
        MODELS.register_module(name="DummyWrapper", module=DummyWrapper)
        MODELS.register_module(name="WhiteNoise", module=WhiteNoise)
        MODELS.register_module(name="TimeSteps", module=TimeSteps)
        MODELS.register_module(name="WuerstchenRandomTimeSteps",
                               module=WuerstchenRandomTimeSteps)
        return super().setUp()

    def tearDown(self):
        MODELS.module_dict.pop("DummyWrapper")
        MODELS.module_dict.pop("WhiteNoise")
        MODELS.module_dict.pop("TimeSteps")
        MODELS.module_dict.pop("WuerstchenRandomTimeSteps")
        return super().tearDown()

    def test_init(self):
        PeftSaveHook()

    def test_before_save_checkpoint(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model = Config.fromfile("tests/configs/sd.py").model
        cfg.model.unet_lora_config = dict(
                    type="LoRA", r=4,
                    target_modules=["to_q", "to_v", "to_k", "to_out.0"])
        runner = self.build_runner(cfg)
        checkpoint = dict(
            state_dict=MODELS.build(cfg.model).state_dict())
        hook = PeftSaveHook()
        hook.before_save_checkpoint(runner, checkpoint)

        assert Path(
            osp.join(runner.work_dir, f"step{runner.iter}/unet",
                     "adapter_model.safetensors")).exists()
        shutil.rmtree(
            osp.join(runner.work_dir, f"step{runner.iter}"))

        for key in checkpoint["state_dict"]:
            assert key.startswith(("unet", "text_encoder"))
            assert "default" in key

    def test_before_save_checkpoint_text_encoder(self):
        # with text encoder
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model = Config.fromfile("tests/configs/sd.py").model
        cfg.model.unet_lora_config = dict(
                    type="LoRA", r=4,
                    target_modules=["to_q", "to_v", "to_k", "to_out.0"])
        cfg.model.text_encoder_lora_config = dict(
                    type="LoRA", r=4,
                    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"])
        cfg.model.finetune_text_encoder = True
        runner = self.build_runner(cfg)
        checkpoint = dict(
            state_dict=MODELS.build(cfg.model).state_dict())
        hook = PeftSaveHook()
        hook.before_save_checkpoint(runner, checkpoint)

        assert Path(
            osp.join(runner.work_dir, f"step{runner.iter}/unet",
                     "adapter_model.safetensors")).exists()
        assert Path(
            osp.join(runner.work_dir, f"step{runner.iter}/text_encoder",
                     "adapter_model.safetensors")).exists()
        shutil.rmtree(
            osp.join(runner.work_dir, f"step{runner.iter}"))

        for key in checkpoint["state_dict"]:
            assert key.startswith(("unet", "text_encoder"))
            assert "default" in key

        # sdxl
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model = Config.fromfile("tests/configs/sdxl.py").model
        cfg.model.unet_lora_config = dict(
                    type="LoRA", r=4,
                    target_modules=["to_q", "to_v", "to_k", "to_out.0"])
        cfg.model.text_encoder_lora_config = dict(
                    type="LoRA", r=4,
                    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"])
        cfg.model.finetune_text_encoder = True
        runner = self.build_runner(cfg)
        checkpoint = dict(
            state_dict=MODELS.build(cfg.model).state_dict())
        hook = PeftSaveHook()
        hook.before_save_checkpoint(runner, checkpoint)

        assert Path(
            osp.join(runner.work_dir, f"step{runner.iter}/unet",
                     "adapter_model.safetensors")).exists()
        assert Path(
            osp.join(runner.work_dir, f"step{runner.iter}/text_encoder_one",
                     "adapter_model.safetensors")).exists()
        assert Path(
            osp.join(runner.work_dir, f"step{runner.iter}/text_encoder_two",
                     "adapter_model.safetensors")).exists()
        shutil.rmtree(
            osp.join(runner.work_dir, f"step{runner.iter}"))

        for key in checkpoint["state_dict"]:
            assert key.startswith(("unet", "text_encoder"))
            assert "default" in key

    def test_before_save_checkpoint_wuerstchen(self):
        # with text encoder
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model = Config.fromfile("tests/configs/wuerstchen.py").model
        cfg.model.prior_lora_config = dict(
                    type="LoRA", r=4,
                    target_modules=["to_q", "to_v", "to_k", "to_out.0"])
        cfg.model.text_encoder_lora_config = dict(
                    type="LoRA", r=4,
                    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"])
        cfg.model.finetune_text_encoder = True
        runner = self.build_runner(cfg)
        checkpoint = dict(
            state_dict=MODELS.build(cfg.model).state_dict())
        hook = PeftSaveHook()
        hook.before_save_checkpoint(runner, checkpoint)

        assert Path(
            osp.join(runner.work_dir, f"step{runner.iter}/prior",
                     "adapter_model.safetensors")).exists()
        assert Path(
            osp.join(runner.work_dir, f"step{runner.iter}/text_encoder",
                     "adapter_model.safetensors")).exists()
        shutil.rmtree(
            osp.join(runner.work_dir, f"step{runner.iter}"))

        for key in checkpoint["state_dict"]:
            assert key.startswith(("prior", "text_encoder"))
            assert "default" in key
