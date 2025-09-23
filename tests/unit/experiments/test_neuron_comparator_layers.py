import src.models.bootstrap  # noqa: F401
from src.models.architectures.configs.mlp import MLPConfig
from src.models.architectures.model_factory import create_model

def _make_mlp(**kwargs):
    cfg = MLPConfig(
        input_dim=3,
        hidden_dims=[2],
        activation="relu",
        output_dim=1,
        start_activation=False,
        end_activation=False,
        **kwargs,
    )
    model = create_model(cfg)
    return model

def test_extract_weights_handles_custom_linear_layers(monkeypatch):
    from src.experiments.utilities.neuron_comparator import NeuronComparator
    comparator = NeuronComparator.__new__(NeuronComparator)
    teacher = _make_mlp()
    student = _make_mlp(mup=True)
    t_hidden, t_out, t_bias = comparator._extract_weights(teacher)
    s_hidden, s_out, s_bias = comparator._extract_weights(student)
    assert t_hidden.shape == (2, 3)
    assert s_hidden.shape == (2, 3)
    assert t_out.shape == (2,)
    assert s_out.shape == (2,)
    assert t_bias.shape == (2,)
    assert s_bias.shape == (2,)

