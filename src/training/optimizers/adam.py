from mup import MuAdam, set_base_shapes

from src.models.mlp import MLP
from src.training.optimizers.configs.adam import AdamConfig
from src.training.optimizers.optimizer import Optimizer, NullStepper
from src.training.optimizers.optimizer_registry import register_optimizer

@register_optimizer("Adam")
class Adam(Optimizer):
    def __init__(self, config: AdamConfig, model: MLP) -> None:
        super().__init__(config, model)

        # make sure we got the right config
        assert isinstance(self.config, AdamConfig)

        param_groups = self._parameter_groups()
        if not param_groups:
            self.stepper = NullStepper()
            return

        assert (
            self.config.mup
        ), "Adam optimizer now assumes μP training; set mup=True in the config."

        base_model = self.model.get_base_model()
        rescale = not getattr(self.model, "mup", False)
        set_base_shapes(self.model, base_model, rescale_params=rescale)
        self.stepper = MuAdam(
            param_groups, lr=self.config.lr, weight_decay=self.config.weight_decay
        )
