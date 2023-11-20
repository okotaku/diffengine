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
        mode (str): The mode to use for compilation. Defaults to None.
        compile_unet (bool): Whether to compile the unet. Defaults to False.
    """

    priority = "VERY_LOW"

    def __init__(self, backend: str = "inductor", mode: str | None = None, *,
                 compile_unet: bool = False) -> None:
        super().__init__()
        self.backend = backend
        self.mode = mode
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
            if hasattr(model, "_forward_compile"):
                # controlnet / t2i adapter
                target = "_forward_compile"
                func = getattr(model, target)
                compiled_func = torch.compile(
                    func, backend=self.backend, mode=self.mode)
                setattr(model, target, compiled_func)
            else:
                model.unet = torch.compile(model.unet, backend=self.backend,
                                        mode=self.mode)

        if hasattr(model, "text_encoder"):
            model.text_encoder = torch.compile(
                model.text_encoder, backend=self.backend, mode=self.mode)
        if hasattr(model, "text_encoder_one"):
            model.text_encoder_one = torch.compile(
                model.text_encoder_one, backend=self.backend, mode=self.mode)
        if hasattr(model, "text_encoder_two"):
            model.text_encoder_two = torch.compile(
                model.text_encoder_two, backend=self.backend, mode=self.mode)
        if hasattr(model, "vae"):
            model.vae = torch.compile(
                model.vae, backend=self.backend, mode=self.mode)
