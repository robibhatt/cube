from pathlib import Path
import pytest

@pytest.fixture(scope="session")
def benchmark_logger():
    """Return a logger for writing benchmark results.

    The file is overwritten on each test session to only keep the latest
    results.  Each log entry contains the iterator name, the underlying
    distribution, batch size, dataset size and number of epochs that were
    timed.  Columns are padded for easy comparison between entries.
    """

    results_path = Path(__file__).parent / "results.txt"
    f = results_path.open("w")

    def describe_target_function(target):
        cfg = target.config.to_dict()
        cfg.pop("model_type", None)
        cfg.pop("input_shape", None)
        cfg.pop("output_shape", None)
        params = ", ".join(f"{k}={v}" for k, v in cfg.items())
        return (
            f"{target.__class__.__name__}({params})"
            if params
            else target.__class__.__name__
        )

    def describe_model(cfg):
        return (
            f"{cfg.model_type}(hidden={cfg.hidden_dims}, activation={cfg.activation})"
        )

    def describe_distribution(dist):
        if dist.config.distribution_type == "CubeDistribution":
            base = getattr(
                dist,
                "base_distribution_description",
                "UniformHypercube",
            )
            noise = getattr(
                dist,
                "noise_distribution_description",
                f"Normal(mean={dist.config.noise_mean}, std={dist.config.noise_std})",
            )
            return f"CubeDistribution(base={base}, noise={noise})"
        return dist.__class__.__name__

    def log(iterator, distribution, batch_size, dataset_size, epochs, elapsed):
        dist_desc = describe_distribution(distribution)
        line = (
            f"{iterator:<20} | {dist_desc:<80} | "
            f"batch={batch_size:<4d} | data={dataset_size:<5d} | "
            f"epochs={epochs:<2d} | time={elapsed:.4f}s\n"
        )
        f.write(line)
        f.flush()

    yield log

    f.close()
