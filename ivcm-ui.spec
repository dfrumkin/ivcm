# PyInstaller spec for the `ivcm-ui` Windows deliverable (one-folder, no console).
#
# Built by .github/workflows/build-windows.yml. The workflow first unpacks the models release asset
# into `<repo>/models/{cls,seg}` (each holding `best-weights.ckpt` + `.hydra/config.yaml`); this
# spec bundles those under `models/cls` and `models/seg` so the frozen app (see
# `ivcm.app.config.build_config`) finds them via `sys._MEIPASS`.
#
# See plans/current.md, decisions 3/3a/5.

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

_ROOT = Path(SPECPATH)  # noqa: F821 (SPECPATH is injected by PyInstaller)
_MODELS = _ROOT / "models"
_ICON = _ROOT / "src" / "ivcm" / "app" / "assets" / "icons" / "icon.ico"

# Packages that rely on dynamic imports and/or ship data files PyInstaller misses by default.
_DYNAMIC_PACKAGES = [
    "ivcm",
    "timm",
    "segmentation_models_pytorch",
    "monai",
    "albumentations",
    "lightning",
    "pytorch_lightning",
    "hydra",
    "omegaconf",
    "scipy",
    "skimage",
    "nibabel",
    "torchvision",
]

hiddenimports: list[str] = []
for _pkg in _DYNAMIC_PACKAGES:
    hiddenimports += collect_submodules(_pkg)

datas: list[tuple[str, str]] = []
for _pkg in _DYNAMIC_PACKAGES:
    datas += collect_data_files(_pkg)


def _tree(src_dir: Path, dest_prefix: str) -> list[tuple[str, str]]:
    """Every file under `src_dir` mapped to `dest_prefix/<relative path>` (dotfiles included)."""
    entries: list[tuple[str, str]] = []
    for path in src_dir.rglob("*"):
        if path.is_file():
            dest = Path(dest_prefix) / path.relative_to(src_dir).parent
            entries.append((str(path), str(dest)))
    return entries


# Bundle the two model directories (checkpoint + .hydra/config.yaml) as models/cls and models/seg.
datas += _tree(_MODELS / "cls", "models/cls")
datas += _tree(_MODELS / "seg", "models/seg")

a = Analysis(  # noqa: F821
    ["src/ivcm/app/__main__.py"],
    pathex=["src"],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["wandb"],  # experiment tracking is training-only; keep it out of the app bundle
    noarchive=False,
)

pyz = PYZ(a.pure)  # noqa: F821

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ivcm-ui",
    console=False,  # windowed app: no console window on double-click
    icon=str(_ICON),
)

coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    name="ivcm-ui",
)
