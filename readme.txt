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
Functions such as ``CubeDistribution.sample`` require a ``torch.Generator``.
Instantiate distributions directly by constructing
``CubeDistributionConfig`` and passing it to ``CubeDistribution``.

```python
from pathlib import Path
import torch
from src.data.joint_distributions.cube_distribution import CubeDistribution
from src.data.joint_distributions.configs.cube_distribution import CubeDistributionConfig
from src.models.targets.sum_prod import SumProdTarget
from src.models.architectures.configs.mlp import MLPConfig
from src.training.optimizers.configs.adam import AdamConfig
from src.training.trainer_config import TrainerConfig
from src.training.trainer import Trainer

cube_cfg = CubeDistributionConfig(
    input_dim=3,
    indices_list=[[0], [1], [2]],
    weights=[1.0, 1.0, 1.0],
    noise_mean=0.0,
    noise_std=0.0,
)
cfg = TrainerConfig(
    mlp_config=MLPConfig(
        input_dim=cube_cfg.input_dim,
        output_dim=1,
        hidden_dims=[4],
        activation="relu",
        start_activation=False,
        end_activation=False,
    ),
    optimizer_config=AdamConfig(lr=1e-3),
    cube_distribution_config=cube_cfg,
    training_size=32,
    test_size=16,
    batch_size=8,
    epochs=1,
    home_dir=Path("/tmp/home"),
    seed=0,
)

trainer = Trainer(cfg)
target = SumProdTarget(cube_cfg.target_function_config)
# To construct a μP-scaled network instead of the standard variant, set
# ``cfg.mlp_config.mup = True`` before creating the model.
gen = torch.Generator(device=trainer.device)
dist = CubeDistribution(cube_cfg, trainer.device)
provider = create_data_provider_from_distribution(
    dist,
    Path("/tmp"),
    batch_size=32,
    dataset_size=100,
    seed=0,
)
X, y = dist.sample(16, seed=0)
```
