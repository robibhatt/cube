from mup import MuSGD, set_base_shapes
from torch.optim import SGD as TorchSGD

from src.models.architectures.model import Model
from src.training.optimizers.configs.sgd import SgdConfig
from src.training.optimizers.optimizer import Optimizer, NullStepper
from src.training.optimizers.optimizer_registry import register_optimizer


@register_optimizer("sgd")
class Sgd(Optimizer):
    def __init__(self, config: SgdConfig, model: Model) -> None:
        super().__init__(config, model)

        assert isinstance(self.config, SgdConfig)

        param_groups = self._parameter_groups()
        if not param_groups:
            self.stepper = NullStepper()
        elif self.config.mup:
            base_model = self.model.get_base_model()
            rescale = not getattr(self.model, "mup", False)
            set_base_shapes(self.model, base_model, rescale_params=rescale)
            self.stepper = MuSGD(
                param_groups,
                lr=self.config.lr,
                momentum=0.0,
                weight_decay=self.config.weight_decay,
            )
        else:
            self.stepper = TorchSGD(
                param_groups,
                lr=self.config.lr,
                momentum=0.0,
                weight_decay=self.config.weight_decay,
            )

