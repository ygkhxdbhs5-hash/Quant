"""Google Colab one-shot: clear cache + full NASDAQ data refresh.

Usage in Colab (one cell):

    !git clone --branch cursor/full-nasdaq-universe-cb1c https://github.com/ygkhxdbhs5-hash/Quant.git
    %cd Quant
    !pip -q install -r requirements.txt
    # then upload/run this file, or:
    %run notebooks/colab_update_data.py

Or from a notebook: import / exec this module after setting MASSIVE_API_KEY.
"""

from __future__ import annotations

import os
import pickle
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# ====== USER SETTINGS ======
REPO_DIR = Path(os.environ.get("QUANT_REPO_DIR", Path.cwd()))
CONFIG = "config/config.yaml"
CLEAR_CACHE = True
RUN_UNIVERSE = True
RUN_PRICES = True
RUN_FUNDAMENTALS = True
SAVE_TO_DRIVE = False
DRIVE_DATA_DIR = Path("/content/drive/MyDrive/Quant_data")


def _resolve_api_key() -> str:
    key = os.environ.get("MASSIVE_API_KEY") or os.environ.get("POLYGON_API_KEY") or ""
    if key:
        return key
    try:
        from google.colab import userdata  # type: ignore

        key = userdata.get("MASSIVE_API_KEY") or ""
    except Exception:
        key = ""
    if not key:
        import getpass

        key = getpass.getpass("MASSIVE_API_KEY: ").strip()
    if not key:
        raise RuntimeError("MASSIVE_API_KEY is required")
    return key


def clear_cache(cache_dir: Path) -> int:
    cache_dir.mkdir(exist_ok=True)
    removed = 0
    for p in cache_dir.rglob("*"):
        if p.is_file() and p.name != ".gitkeep":
            p.unlink(missing_ok=True)
            removed += 1
    for p in sorted(cache_dir.rglob("*"), reverse=True):
        if p.is_dir() and p != cache_dir:
            try:
                p.rmdir()
            except OSError:
                pass
    (cache_dir / "massive_cache").mkdir(parents=True, exist_ok=True)
    (cache_dir / ".gitkeep").touch()
    return removed


def run_step(title: str, module: str) -> None:
    print("\n" + "=" * 72)
    print(f"[{datetime.now().isoformat(timespec='seconds')}] START {title}")
    print("=" * 72)
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.run(
        [sys.executable, "-u", "-m", module, "--config", CONFIG],
        cwd=str(REPO_DIR),
        env=env,
    )
    print(f"[{datetime.now().isoformat(timespec='seconds')}] END {title} exit={proc.returncode}")
    if proc.returncode != 0:
        raise RuntimeError(f"{module} failed with exit {proc.returncode}")


def verify() -> None:
    uni_path = REPO_DIR / "data/metadata/universe.pkl"
    px_path = REPO_DIR / "data/prices/panels.pkl"
    fund_path = REPO_DIR / "data/fundamentals/pit_history.pkl"

    uni = pickle.load(open(uni_path, "rb"))
    tickers = uni.get("tickers") or []
    all_tickers = uni.get("all_tickers") or []
    profiles = uni.get("profile_meta") or {}
    print(f"all_tickers   : {len(all_tickers)}")
    print(f"tickers       : {len(tickers)}")
    print(f"profile_meta  : {len(profiles)}")
    print(f"tickers==all? : {tickers == all_tickers}")
    if len(tickers) < 1000:
        print("WARNING: tickers < 1000 — may not be full NASDAQ universe")
    else:
        print("OK: universe size looks like a full NASDAQ pull")

    if px_path.exists():
        panels = pickle.load(open(px_path, "rb"))
        close_m = panels["close_m"]
        print(f"prices        : {close_m.shape}")
    if fund_path.exists():
        funds = pickle.load(open(fund_path, "rb"))
        print(f"fundamentals  : {len(funds)} symbols")


def backup_to_drive() -> None:
    if not SAVE_TO_DRIVE:
        return
    try:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive", force_remount=False)
    except Exception as exc:
        print("Drive mount skipped/failed:", exc)
        return

    DRIVE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("data", "cache"):
        src = REPO_DIR / name
        dst = DRIVE_DATA_DIR / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"copied {src} -> {dst}")


def main() -> None:
    os.chdir(REPO_DIR)
    key = _resolve_api_key()
    os.environ["MASSIVE_API_KEY"] = key
    os.environ["PYTHONUNBUFFERED"] = "1"
    print("API key set:", key[:4] + "…" + key[-4:])
    print("repo:", REPO_DIR.resolve())

    if CLEAR_CACHE:
        n = clear_cache(REPO_DIR / "cache")
        print(f"Cleared cache files: {n}")

    if RUN_UNIVERSE:
        run_step("1/3 universe", "downloader.download_universe")
    if RUN_PRICES:
        run_step("2/3 prices", "downloader.download_prices")
    if RUN_FUNDAMENTALS:
        run_step("3/3 fundamentals", "downloader.download_fundamentals")

    verify()
    backup_to_drive()
    print("\nDone.")


if __name__ == "__main__":
    main()
