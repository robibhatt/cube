to do stuff run
    conda activate robi_cluster_env
    pip install -r requirements.txt
    sbatch submit_job.sh

also todo one time: (the pip installations)


just need to keep requirements up to date.
pip freeze > requirements.txt

to check progress on job do this:

squeue -u luj210

to login:

ssh luj210@galvani-login.mlcloud.uni-tuebingen.de
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

Running tests
-------------
Use `pytest` to run all tests. Some tests are marked `integration`.

* `pytest` runs everything.
* `pytest -m "not integration"` runs unit tests only.
* `pytest -m integration` runs integration tests only.

Specifying devices
------------------
Pass a ``device`` argument when creating a :class:`Trainer`.  Provide a
:class:`torch.device` instance such as ``torch.device("cpu")`` or
``torch.device("cuda:0")`` to select where tensors are allocated.
Functions such as ``JointDistribution.sample`` require a ``torch.Generator``.
Distributions themselves are instantiated via the ``create_joint_distribution``
factory function, which looks up classes in registry dictionaries.

```python
from pathlib import Path
import torch
from src.data.joint_distributions.configs.gaussian import GaussianConfig
from src.data.joint_distributions.joint_distribution_factory import create_joint_distribution
from src.models.targets.sum_prod import SumProdTarget
from src.models.targets.configs.sum_prod import SumProdTargetConfig
from src.data.joint_distributions.configs.mapped_joint_distribution import (
    MappedJointDistributionConfig,
)
from src.models.architectures.configs.mlp import MLPConfig
from src.training.optimizers.configs.adam import AdamConfig
from src.training.trainer_config import TrainerConfig
from src.training.loss.configs.loss import LossConfig
from src.training.trainer import Trainer

dist_cfg = GaussianConfig(input_shape=torch.Size([2]), mean=0.0, std=1.0)
dist = create_joint_distribution(dist_cfg, device=torch.device("cpu"))
target_cfg = SumProdTargetConfig(
    input_shape=dist.input_shape,
    indices_list=[[0, 1]],
    weights=[1.0],
)
joint_cfg = MappedJointDistributionConfig(
    distribution_config=dist_cfg,
    target_function_config=target_cfg,
)
cfg = TrainerConfig(
    model_config=MLPConfig(
        input_dim=2,
        output_dim=1,
        hidden_dims=[4],
        activation="relu",
        start_activation=False,
        end_activation=False,
    ),
    optimizer_config=AdamConfig(lr=1e-3),
    joint_distribution_config=joint_cfg,
    training_size=32,
    test_size=16,
    batch_size=8,
    epochs=1,
    home_dir=Path("/tmp/home"),
    loss_config=LossConfig(name="MSELoss"),
    seed=0,
)

trainer = Trainer(cfg)
target = SumProdTarget(target_cfg)
# To construct a μP-scaled network instead of the standard variant, set
# ``cfg.model_config.mup = True`` before creating the model.
gen = torch.Generator(device=trainer.device)
provider = create_data_provider_from_distribution(
    dist,
    Path("/tmp"),
    batch_size=32,
    dataset_size=100,
    seed=0,
)
X, y = dist.sample(16, seed=0)
```
