from mmengine.hooks.hook import DATA_BATCH, Hook
from mmengine.model import is_model_wrapper
from mmengine.registry import HOOKS


@HOOKS.register_module()
class LCMEMAUpdateHook(Hook):
    """LCM EMA Update Hook."""

    def before_run(self, runner) -> None:
        """Create an ema copy of the model.

        Args:
        ----
            runner (Runner): The runner of the training process.
        """
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        self.src_model = model.unet
        self.ema_model = model.target_unet

    def after_train_iter(self,
                         runner,  # noqa
                         batch_idx: int,  # noqa
                         data_batch: DATA_BATCH = None,  # noqa
                         outputs: dict | None = None) -> None:  # noqa
        """Update ema parameter.

        Args:
        ----
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (Sequence[dict], optional): Data from dataloader.
                Defaults to None.
            outputs (dict, optional): Outputs from model. Defaults to None.
        """
        self.ema_model.update_parameters(self.src_model)
