from typing import Tuple, TYPE_CHECKING

import torch

from .joint_distribution import JointDistribution
from .configs.paired_representor_distribution import (
    PairedRepresentorDistributionConfig,
)
from .joint_distribution_registry import register_joint_distribution

if TYPE_CHECKING:
    from src.data.providers.data_provider import DataProvider


@register_joint_distribution("PairedRepresentorDistribution")
class PairedRepresentorDistribution(JointDistribution):
    def __init__(
        self, config: PairedRepresentorDistributionConfig, device: torch.device
    ) -> None:
        from .joint_distribution_factory import create_joint_distribution
        from src.models.representors.representor_factory import (
            create_model_representor,
        )

        self.base_joint_distribution = create_joint_distribution(
            config.base_distribution_config, device
        )
        self.teacher_representor = create_model_representor(
            config.teacher_model_config,
            config.teacher_checkpoint_dir,
            device=device,
        )
        self.student_representor = create_model_representor(
            config.student_model_config,
            config.student_checkpoint_dir,
            device=device,
        )
        self.teacher_rep_id = config.teacher_rep_id
        self.student_rep_id = config.student_rep_id
        self.teacher_from_rep_id = config.teacher_from_rep_id
        self.student_from_rep_id = config.student_from_rep_id

        super().__init__(config=config, device=device)

    def __str__(self) -> str:
        return (
            f"PairedRepresentorDistribution(base_distribution={self.base_joint_distribution}, "
            f"student_representor={self.student_representor}, student_from_rep_id={self.student_from_rep_id}, student_rep_id={self.student_rep_id}, "
            f"teacher_representor={self.teacher_representor}, teacher_from_rep_id={self.teacher_from_rep_id}, teacher_rep_id={self.teacher_rep_id})"
        )

    def sample(self, n_samples: int, seed: int):
        X_base, _ = self.base_sample(n_samples, seed=seed)
        return self.forward(X_base)

    def base_sample(self, n_samples: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.base_joint_distribution.base_sample(n_samples, seed)

    def forward(self, base_X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        X_penultimate = self.base_joint_distribution.forward_X(base_X)
        student_out = self.student_representor.forward(
            X_penultimate, self.student_from_rep_id, self.student_rep_id, None
        )[1]
        teacher_out = self.teacher_representor.forward(
            X_penultimate, self.teacher_from_rep_id, self.teacher_rep_id, None
        )[1]
        return student_out, teacher_out

    def forward_X(self, base_X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward(base_X)

    def preferred_provider(self) -> str:
        return "TensorDataProvider"
