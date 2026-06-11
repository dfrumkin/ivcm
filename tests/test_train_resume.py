"""Tests for W&B run-id continuation on resume.

Covers the shared `stable_wandb_id` helper and train.run()'s `_wandb_anchor_dir` /
`_pin_wandb_id`: a resumed run continues the same W&B run by anchoring a persisted run id to
its checkpoint directory, while a fresh run (with a stable pinned dir) gets a new one. These
are white-box unit tests of train.py's private helpers, so the private-use warnings on the
imported names are suppressed."""

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from ivcm.train import (
    _pin_wandb_id,  # pyright: ignore[reportPrivateUsage]
    _wandb_anchor_dir,  # pyright: ignore[reportPrivateUsage]
)
from ivcm.utils import stable_wandb_id

CONFIGS_DIR = str(Path(__file__).resolve().parents[1] / "configs")

WANDB_TARGET = "lightning.pytorch.loggers.WandbLogger"


def _cfg(
    *,
    ckpt_path: str | None = None,
    callbacks: list[str] | None = None,
    logger_target: str | None = WANDB_TARGET,
    logger_id: str | None = None,
) -> DictConfig:
    """A struct-mode cfg with just the keys the helpers read (mirrors the real Hydra config's
    struct flag so the `open_dict` path in `_pin_wandb_id` is exercised)."""
    logger: dict[str, str] | None = None
    if logger_target is not None:
        logger = {"_target_": logger_target}
        if logger_id is not None:
            logger["id"] = logger_id
    data: dict[str, object] = {"ckpt_path": ckpt_path, "callbacks": callbacks, "logger": logger}
    cfg = OmegaConf.create(data)
    OmegaConf.set_struct(cfg, True)
    return cfg


def test_stable_wandb_id_is_stable(tmp_path: Path) -> None:
    """The id is generated once, persisted, and reloaded (so a resume continues the same run)."""
    directory = tmp_path / "checkpoints"
    first = stable_wandb_id(directory)
    assert first and (directory / "wandb_run_id.txt").exists()
    assert stable_wandb_id(directory) == first  # reloaded, not regenerated


def test_anchor_resume_continues_regardless_of_callbacks(tmp_path: Path) -> None:
    """Resuming from a real checkpoint anchors to its own dir even under `callbacks=none` — the
    case that was previously skipped. The new run_dir is irrelevant on resume."""
    ckpt = tmp_path / "orig" / "checkpoints" / "last.ckpt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_text("ckpt")
    cfg = _cfg(ckpt_path=str(ckpt), callbacks=None)
    assert _wandb_anchor_dir(cfg, str(tmp_path / "new")) == ckpt.resolve().parent


def test_anchor_fresh_callbacks_none_has_no_anchor(tmp_path: Path) -> None:
    """A fresh `callbacks=none` run has no stable pinned checkpoint dir, so no anchor to tag."""
    cfg = _cfg(ckpt_path=None, callbacks=None)
    assert _wandb_anchor_dir(cfg, str(tmp_path)) is None


def test_anchor_fresh_callbacks_default_uses_run_checkpoints(tmp_path: Path) -> None:
    """A fresh run with callbacks set anchors to <run_dir>/checkpoints (where it pins the
    ModelCheckpoint)."""
    cfg = _cfg(ckpt_path=None, callbacks=["cb"])
    assert _wandb_anchor_dir(cfg, str(tmp_path)) == (tmp_path / "checkpoints").resolve()


def test_pin_fresh_run_injects_id(tmp_path: Path) -> None:
    """A WandbLogger config with a fresh anchor gets a generated id, resume='allow', and a
    persisted sidecar."""
    cfg = _cfg()
    directory = tmp_path / "checkpoints"
    _pin_wandb_id(cfg, directory)
    assert cfg.logger.id
    assert cfg.logger.resume == "allow"
    assert (directory / "wandb_run_id.txt").read_text().strip() == cfg.logger.id


def test_pin_true_resume_reuses_id(tmp_path: Path) -> None:
    """Anchoring at a dir that already holds a sidecar continues that run (reuses its id)."""
    directory = tmp_path / "checkpoints"
    directory.mkdir()
    (directory / "wandb_run_id.txt").write_text("fixedid123")
    cfg = _cfg()
    _pin_wandb_id(cfg, directory)
    assert cfg.logger.id == "fixedid123"


def test_pin_untagged_dirs_get_distinct_ids(tmp_path: Path) -> None:
    """Two fresh, untagged dirs get independent new ids (i.e. distinct W&B runs)."""
    cfg_a, cfg_b = _cfg(), _cfg()
    _pin_wandb_id(cfg_a, tmp_path / "a")
    _pin_wandb_id(cfg_b, tmp_path / "b")
    assert cfg_a.logger.id and cfg_b.logger.id
    assert cfg_a.logger.id != cfg_b.logger.id


def test_pin_repeated_resume_from_untagged_dir_continues(tmp_path: Path) -> None:
    """Dir-identity regression: two resumes from the SAME untagged dir continue, not fork. The
    first tags the dir; the second reuses that id."""
    directory = tmp_path / "checkpoints"
    cfg_first, cfg_second = _cfg(), _cfg()
    _pin_wandb_id(cfg_first, directory)
    _pin_wandb_id(cfg_second, directory)
    assert cfg_first.logger.id == cfg_second.logger.id
    assert (directory / "wandb_run_id.txt").exists()


def test_pin_does_not_overwrite_existing_id(tmp_path: Path) -> None:
    """An already-set id (cv.py pins its own per-fold id before run()) always wins; no sidecar
    is written."""
    cfg = _cfg(logger_id="cvfold0")
    directory = tmp_path / "checkpoints"
    _pin_wandb_id(cfg, directory)
    assert cfg.logger.id == "cvfold0"
    assert not (directory / "wandb_run_id.txt").exists()


def test_pin_no_op_without_wandb_logger(tmp_path: Path) -> None:
    """Disabled (`logger=none`) and non-WandbLogger configs are left untouched, no sidecar."""
    cfg_none = _cfg(logger_target=None)
    _pin_wandb_id(cfg_none, tmp_path / "a")
    assert cfg_none.get("logger") is None

    cfg_tb = _cfg(logger_target="lightning.pytorch.loggers.TensorBoardLogger")
    _pin_wandb_id(cfg_tb, tmp_path / "b")
    assert "id" not in cfg_tb.logger

    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()


def test_pin_no_op_without_anchor(tmp_path: Path) -> None:
    """No anchor (fresh callbacks=none) -> the WandbLogger keeps its default new-run behavior."""
    cfg = _cfg()
    _pin_wandb_id(cfg, None)
    assert "id" not in cfg.logger


def test_non_cv_segmentation_train_run_gets_id(tmp_path: Path) -> None:
    """Scope guard: continuation is not classification-only. A single (non-CV) segmentation
    `train` run with callbacks=default also gets an injected id, proving no task gating."""
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = compose(config_name="train", overrides=["task=segmentation", "callbacks=default"])

    wandb_dir = _wandb_anchor_dir(cfg, str(tmp_path))
    assert wandb_dir == (tmp_path / "checkpoints").resolve()
    _pin_wandb_id(cfg, wandb_dir)
    assert cfg.logger.id
    assert cfg.logger.resume == "allow"
