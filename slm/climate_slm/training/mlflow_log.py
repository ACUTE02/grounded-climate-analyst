"""MLflow logging with a local file store (outputs/mlruns, no server). mlflow is imported only inside the functions,
so the tests and the laptop never need it.

Two rules:
  * Newer MLflow versions switch the file store off unless MLFLOW_ALLOW_FILE_STORE=true. It is set here, at import of
    this module and so before mlflow is imported (a value the user already set is kept).
  * An MLflow error must never stop a training or an evaluation. Every call that talks to MLflow catches the error,
    prints a warning and goes on. Training errors in the train error file are therefore never MLflow errors."""
import os

from climate_slm.config import REPO_ROOT

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")


def warn(what: str, err: BaseException) -> None:
    print(f"WARNING: MLflow {what} failed, logging is skipped and the run goes on: {type(err).__name__}: {err}")


def tracking_uri(cfg3: dict) -> str:
    return (REPO_ROOT / cfg3["paths"]["mlruns_dir"]).resolve().as_uri()


def _mlflow(cfg3: dict):
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")     # also if the module was imported in another order
    import mlflow
    mlflow.set_tracking_uri(tracking_uri(cfg3))
    mlflow.set_experiment(cfg3["mlflow"]["experiment_name"])
    return mlflow


def clean_params(params: dict) -> dict[str, str]:
    """MLflow params are strings of at most 500 characters."""
    return {k: str(v)[:490] for k, v in params.items()}


class Tracker:
    """The MLflow run of one training. If MLflow is missing or broken, `mlflow` is None and every call does nothing."""

    def __init__(self, mlflow=None, run_id: str | None = None):
        self.mlflow, self.run_id = mlflow, run_id

    @property
    def active(self) -> bool:
        return self.mlflow is not None

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if not self.active:
            return
        try:
            self.mlflow.log_metrics(metrics, step=step) if step is not None else self.mlflow.log_metrics(metrics)
        except Exception as err:                                   # noqa: BLE001  (any MLflow problem)
            warn("log_metrics", err)

    def end(self, status: str = "FINISHED") -> None:
        if not self.active:
            return
        try:
            self.mlflow.end_run(status=status)
        except Exception as err:                                   # noqa: BLE001
            warn("end_run", err)


def start_tracker(cfg3: dict, run_name: str, params: dict, tags: dict) -> Tracker:
    """Start a run that stays active while the training logs into it. Never raises: on any MLflow error a
    switched-off Tracker is returned (run_id None) and the training goes on without logging."""
    try:
        mlflow = _mlflow(cfg3)
        run = mlflow.start_run(run_name=run_name)
        tracker = Tracker(mlflow, run.info.run_id)
    except Exception as err:                                       # noqa: BLE001
        warn("start", err)
        return Tracker()
    try:
        mlflow.log_params(clean_params(params))
        mlflow.set_tags({k: str(v) for k, v in tags.items()})
    except Exception as err:                                       # noqa: BLE001  (the run exists, only params are missing)
        warn("log_params", err)
    return tracker


def log_metrics_into_run(cfg3: dict, run_id: str, metrics: dict[str, float], step: int) -> bool:
    """Add metrics to a run that already exists (the per-epoch val evaluation after training).
    Returns False, with a warning, if MLflow failed."""
    try:
        mlflow = _mlflow(cfg3)
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(metrics, step=step)
        return True
    except Exception as err:                                       # noqa: BLE001
        warn("log_metrics_into_run", err)
        return False
