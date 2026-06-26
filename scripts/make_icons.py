"""Regenerate the per-platform app icons from the committed master PNG.

The single source of truth is ``src/ivcm/app/assets/icons/icon.png`` (1024x1024). This script
derives every other committed asset from it, so regeneration needs no external file:

- Windows ``icon.ico``      — one multi-size .ico (16/24/32/48/64/128/256).
- Linux  ``linux/hicolor/`` — square PNGs in the freedesktop hicolor tree
                              (``<size>x<size>/apps/ivcm-ui.png``), matching the .desktop Icon name.
- macOS  ``icon.icns``      — built from a standard .iconset via ``iconutil`` (macOS only).

The master itself was produced once by downscaling the original artwork to 1024x1024; from then on
this script is the only thing that touches the derived assets. It is a developer convenience (run
when the artwork changes); the generated assets are committed, so neither the app nor CI runs it.

Usage:

    uv run python scripts/make_icons.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

ICONS_DIR = Path(__file__).resolve().parents[1] / "src" / "ivcm" / "app" / "assets" / "icons"
MASTER = ICONS_DIR / "icon.png"

# Windows .ico embeds these square sizes; Linux hicolor ships the same standard sizes as PNGs.
ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)
HICOLOR_SIZES = (16, 32, 48, 64, 128, 256, 512)

# macOS .iconset entries: (filename, pixel size). The @2x variants are retina (2x the point size).
ICONSET_ENTRIES = (
    ("icon_16x16.png", 16),
    ("icon_16x16@2x.png", 32),
    ("icon_32x32.png", 32),
    ("icon_32x32@2x.png", 64),
    ("icon_128x128.png", 128),
    ("icon_128x128@2x.png", 256),
    ("icon_256x256.png", 256),
    ("icon_256x256@2x.png", 512),
    ("icon_512x512.png", 512),
    ("icon_512x512@2x.png", 1024),
)


def _resized(master: Image.Image, size: int) -> Image.Image:
    return master.resize((size, size), Image.LANCZOS)


def _make_ico(master: Image.Image) -> None:
    out = ICONS_DIR / "icon.ico"
    master.save(out, format="ICO", sizes=[(s, s) for s in ICO_SIZES])
    print(f"wrote {out.relative_to(ICONS_DIR.parents[4])} ({', '.join(str(s) for s in ICO_SIZES)})")


def _make_hicolor(master: Image.Image) -> None:
    base = ICONS_DIR / "linux" / "hicolor"
    for size in HICOLOR_SIZES:
        target = base / f"{size}x{size}" / "apps" / "ivcm-ui.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        _resized(master, size).save(target, format="PNG")
    print(f"wrote linux/hicolor/<size>/apps/ivcm-ui.png ({', '.join(str(s) for s in HICOLOR_SIZES)})")


def _make_icns(master: Image.Image) -> None:
    out = ICONS_DIR / "icon.icns"
    if shutil.which("iconutil") is None:
        message = (
            "WARNING: 'iconutil' not found (non-macOS host); skipping icon.icns. "
            "macOS is the development platform, so the committed icon.icns is now STALE relative "
            "to icon.png — regenerate it on a Mac before relying on the macOS icon."
        )
        print(message, file=sys.stderr)
        return
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "icon.iconset"
        iconset.mkdir()
        for name, size in ICONSET_ENTRIES:
            _resized(master, size).save(iconset / name, format="PNG")
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(out)], check=True)
    print(f"wrote {out.name}")


def main() -> int:
    if not MASTER.exists():
        print(f"ERROR: master icon not found: {MASTER}", file=sys.stderr)
        return 1
    master = Image.open(MASTER).convert("RGBA")
    if master.size != (1024, 1024):
        print(f"ERROR: master must be 1024x1024, got {master.size}", file=sys.stderr)
        return 1
    _make_ico(master)
    _make_hicolor(master)
    _make_icns(master)
    return 0


if __name__ == "__main__":
    sys.exit(main())
