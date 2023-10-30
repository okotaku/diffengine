import torch
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.registry import HOOKS


@HOOKS.register_module()
class CompileHook(Hook):
    """Compile Hook.

    Args:
    ----
        backend (str): The backend to use for compilation.
            Defaults to "inductor".
        compile_unet (bool): Whether to compile the unet. Defaults to False.
    """

    priority = "VERY_LOW"

    def __init__(self, backend: str = "inductor", *,
                 compile_unet: bool = False) -> None:
        super().__init__()
        self.backend = backend
        self.compile_unet = compile_unet

    def before_train(self, runner) -> None:
        """Compile the model.

        Args:
        ----
            runner (Runner): The runner of the training process.
        """
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        if self.compile_unet:
            model.unet = torch.compile(model.unet, backend=self.backend)

        if hasattr(model, "text_encoder"):
            model.text_encoder = torch.compile(
                model.text_encoder, backend=self.backend)
        if hasattr(model, "text_encoder_one"):
            model.text_encoder_one = torch.compile(
                model.text_encoder_one, backend=self.backend)
        if hasattr(model, "text_encoder_two"):
            model.text_encoder_two = torch.compile(
                model.text_encoder_two, backend=self.backend)
        if hasattr(model, "vae"):
            model.vae = torch.compile(
                model.vae, backend=self.backend)